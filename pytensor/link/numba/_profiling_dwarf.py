"""Linux x86-64 perf decoding and recovery through elfutils/libdwfl."""

import ctypes as ct
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from bisect import bisect_right
from collections import Counter, deque
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

from filelock import FileLock

from pytensor.configdefaults import config


_REGISTERS = (0, 3, 2, 1, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8)


@lru_cache(maxsize=1)
def load_unwinder():
    source = Path(__file__).with_name("_profiling_unwind.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(config.compiledir) / "native_profiling"
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f"unwind_{digest}.so"
    with FileLock(str(library) + ".lock"):
        if not library.exists():
            compiler = shutil.which("cc")
            if compiler is None:
                raise RuntimeError(
                    "Native DWARF profiling needs a C compiler and libdw/libelf development packages"
                )
            fd, temporary = tempfile.mkstemp(suffix=".so", dir=directory)
            os.close(fd)
            try:
                result = subprocess.run(
                    [
                        compiler,
                        "-O2",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        temporary,
                        "-ldw",
                        "-lelf",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    (directory / "build.log").write_text(result.stderr)
                    raise RuntimeError(
                        "Cannot build the native unwinder. Install a C compiler and "
                        f"libdw/libelf development packages; see {directory / 'build.log'}. "
                        "Use backend=LinuxPerfBackend() for frame-pointer sampling."
                    )
                Path(temporary).replace(library)
            finally:
                Path(temporary).unlink(missing_ok=True)
    try:
        return ct.CDLL(str(library))
    except OSError as error:
        raise RuntimeError(
            "Cannot load the native profiling helper; libdw/libelf runtime libraries are required"
        ) from error


class _Unwinder:
    def __init__(self, modules):
        library = load_unwinder()

        def bind(name, result, arguments):
            function = getattr(library, "probe_" + name)
            function.restype = result
            function.argtypes = arguments
            return function

        create = bind("create", ct.c_void_p, [])
        add_module = bind("module", ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_uint64])
        attach = bind("attach", ct.c_int, [ct.c_void_p])
        self._unwind = bind(
            "unwind",
            ct.c_int,
            [
                ct.c_void_p,
                ct.POINTER(ct.c_uint64),
                ct.c_uint64,
                ct.c_void_p,
                ct.c_size_t,
                ct.POINTER(ct.c_uint64),
                ct.c_int,
            ],
        )
        self._name = bind("name", ct.c_char_p, [ct.c_void_p, ct.c_uint64])
        self._path = bind("path", ct.c_char_p, [ct.c_void_p, ct.c_uint64])
        self._symbol = bind(
            "symbol", ct.c_int, [ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_uint64)]
        )
        self._close = bind("close", None, [ct.c_void_p])
        self.context = create()
        if not self.context:
            raise RuntimeError("Cannot initialize libdwfl")
        for filename, base in modules:
            add_module(self.context, os.fsencode(filename), base)
        if not attach(self.context):
            self.close()
            raise RuntimeError(
                "Cannot attach the offline DWARF unwinder to recorded modules"
            )

    def close(self):
        if self.context:
            self._close(self.context)
            self.context = None

    def unwind(self, registers, stack_base, stack):
        registers = (ct.c_uint64 * 17)(*registers)
        frames = (ct.c_uint64 * 64)()
        buffer = ct.create_string_buffer(stack)
        count = self._unwind(
            self.context, registers, stack_base, buffer, len(stack), frames, 64
        )
        return list(frames[:count])

    def name(self, pc):
        return (self._name(self.context, pc) or b"[unknown]").decode(errors="replace")

    def path(self, pc):
        return os.fsdecode(self._path(self.context, pc) or b"[unknown]")

    def symbol(self, pc):
        result = (ct.c_uint64 * 3)()
        return tuple(result) if self._symbol(self.context, pc, result) else None


class _Recording:
    def __init__(self, filename):
        self.filename = filename
        self.lost = 0
        with filename.open("rb") as stream:
            header = stream.read(56)
            if len(header) != 56 or header[:8] != b"PERFILE2":
                raise RuntimeError("Unsupported perf recording header")
            _, _, attr_size, attr_offset, attr_length, self.offset, self.length = (
                struct.unpack("<8s6Q", header)
            )
            if attr_size < 104 or attr_length % attr_size:
                raise RuntimeError("Unsupported perf event attributes")
            file_size = filename.stat().st_size
            if (
                self.offset + self.length > file_size
                or attr_offset + attr_length > file_size
            ):
                raise RuntimeError("Truncated perf recording")
            self.attributes = {}
            for position in range(attr_offset, attr_offset + attr_length, attr_size):
                stream.seek(position)
                attribute = stream.read(attr_size)
                sample_type = struct.unpack_from("<Q", attribute, 24)[0]
                register_mask = struct.unpack_from("<Q", attribute, 80)[0]
                ids_offset, ids_length = struct.unpack_from(
                    "<QQ", attribute, attr_size - 16
                )
                if ids_length % 8 or ids_offset + ids_length > file_size:
                    raise RuntimeError("Invalid perf event identifier table")
                stream.seek(ids_offset)
                for (identifier,) in struct.iter_unpack("<Q", stream.read(ids_length)):
                    self.attributes[identifier] = sample_type, register_mask

    def records(self):
        with self.filename.open("rb") as stream:
            stream.seek(self.offset)
            remaining = self.length
            while remaining:
                header = stream.read(8)
                if len(header) != 8:
                    raise RuntimeError("Truncated perf record")
                kind, misc, size = struct.unpack("<IHH", header)
                if size < 8 or size > remaining:
                    raise RuntimeError("Invalid perf record length")
                payload = stream.read(size - 8)
                if len(payload) != size - 8:
                    raise RuntimeError("Truncated perf record payload")
                remaining -= size
                yield kind, misc, payload

    def modules(self):
        modules = set()
        for kind, misc, payload in self.records():
            if kind not in (1, 10):
                continue
            start, _, offset = struct.unpack_from("<3Q", payload, 8)
            if offset:
                continue
            name_offset = 64 if kind == 10 else 32
            path = Path(os.fsdecode(payload[name_offset:].split(b"\0", 1)[0]))
            try:
                with path.open("rb") as stream:
                    magic = stream.read(6)
                if magic == b"\x7fELF\x02\x01":
                    if kind == 10 and not misc & (1 << 14):
                        _, _, inode = struct.unpack_from("<IIQ", payload, 32)
                        status = path.stat()
                        # Device numbers can differ across mount namespaces.
                        if inode and status.st_ino != inode:
                            continue
                    modules.add((str(path), start))
            except OSError:
                continue
        return modules

    def samples(self):
        for kind, _, payload in self.records():
            if kind == 2:
                self.lost += struct.unpack_from("<Q", payload, 8)[0]
            elif kind == 13:
                self.lost += struct.unpack_from("<Q", payload)[0]
            elif kind == 9:
                cursor = 0

                def read(fmt="Q"):
                    nonlocal cursor
                    size = struct.calcsize("<" + fmt)
                    if cursor + size > len(payload):
                        raise RuntimeError("Truncated native sample")
                    result = struct.unpack_from("<" + fmt, payload, cursor)
                    cursor += size
                    return result[0] if len(result) == 1 else result

                identifier = read()
                if identifier not in self.attributes:
                    raise RuntimeError("Unknown perf event identifier")
                sample_type, register_mask = self.attributes[identifier]
                required = (1 << 16) | 1 | (1 << 8) | (1 << 12) | (1 << 13)
                supported = (
                    required
                    | (1 << 1)
                    | (1 << 2)
                    | (1 << 3)
                    | (1 << 5)
                    | (1 << 6)
                    | (1 << 7)
                    | (1 << 9)
                )
                if sample_type & required != required or sample_type & ~supported:
                    raise RuntimeError(
                        f"Unsupported DWARF sample fields: {sample_type:#x}"
                    )
                ip = read()
                for bit in (1, 2, 3, 6, 9, 7):
                    if sample_type & (1 << bit):
                        read()
                weight = read()
                if sample_type & (1 << 5):
                    for _ in range(read()):
                        read()
                abi = read()
                if abi not in (0, 2):
                    raise RuntimeError("DWARF recovery requires x86-64 user registers")
                registers = {}
                if abi:
                    for index in range(64):
                        if register_mask & (1 << index):
                            registers[index] = read()
                    if not set(_REGISTERS) <= registers.keys():
                        raise RuntimeError(
                            "Native recording is missing required user registers"
                        )
                size = read()
                if cursor + size > len(payload):
                    raise RuntimeError("Truncated captured user stack")
                stack = payload[cursor : cursor + size]
                cursor += size
                dynamic_size = read() if size else 0
                if dynamic_size > size:
                    raise RuntimeError("Invalid captured user stack size")
                yield ip, weight, registers, stack[:dynamic_size]


_REG_NAMES = (
    "rax",
    "rcx",
    "rdx",
    "rbx",
    "rsp",
    "rbp",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
)
_REG_INDEX = {name: index for index, name in enumerate(_REG_NAMES)}
_CALLER_SAVED = {
    _REG_INDEX[name]
    for name in ("rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11")
}
_ALIASES = {}
for names in [
    ("rax", "eax", "ax", "al", "ah"),
    ("rcx", "ecx", "cx", "cl", "ch"),
    ("rdx", "edx", "dx", "dl", "dh"),
    ("rbx", "ebx", "bx", "bl", "bh"),
    ("rsp", "esp", "sp", "spl"),
    ("rbp", "ebp", "bp", "bpl"),
    ("rsi", "esi", "si", "sil"),
    ("rdi", "edi", "di", "dil"),
]:
    for name in names:
        _ALIASES[name] = names[0]


def _register(token):
    token = token.lstrip("%")
    if (
        token.startswith("r")
        and token[-1:] in ("d", "w", "b")
        and token[1:-1].isdigit()
    ):
        token = token[:-1]
    return _REG_INDEX.get(_ALIASES.get(token, token))


class _Analysis:
    def __init__(self, directory, external, unwinder):
        self.directory = directory
        self.external = external
        self.unwinder = unwinder
        self.errors = {}
        for name in ("instructions", "fixed_leaf", "call_targets", "is_call_return"):
            setattr(self, name, lru_cache(maxsize=None)(getattr(self, name)))

    def close(self):
        for name in ("instructions", "fixed_leaf", "call_targets", "is_call_return"):
            getattr(self, name).cache_clear()

    def is_call_return(self, pc):
        found = self.unwinder.symbol(pc - 1)
        if found is None:
            return False
        start, size, offset = found
        instructions = self.instructions(self.unwinder.path(pc - 1), offset, size)
        wanted = pc - start + offset
        return any(
            row[1].startswith("call") and following[0] == wanted
            for row, following in pairwise(instructions)
        )

    def instructions(self, filename, start, size, symbol=None):
        filename = str(filename)
        if filename in self.errors:
            return []
        command = ["objdump", "-dr", "--disassemble-zeroes", "--no-show-raw-insn"]
        if symbol:
            command += ["--disassemble=" + symbol]
        else:
            command += [
                "--start-address=" + hex(start),
                "--stop-address=" + hex(start + size),
            ]
        try:
            raw = subprocess.check_output(
                [*command, filename], text=True, stderr=subprocess.PIPE
            )
        except (OSError, subprocess.CalledProcessError) as error:
            self.errors[filename] = (
                getattr(error, "stderr", None) or str(error)
            ).strip()
            return []
        result = []
        for line in raw.splitlines():
            reloc = re.match(r"^\s*([0-9a-f]+):\s+R_X86_64_\w+\s+(\S+)", line)
            if reloc:
                if result:
                    result[-1][3] = reloc[2].split("+", 1)[0]
            elif match := re.match(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line):
                address = int(match[1], 16)
                if start <= address < start + size:
                    operands = match[3].partition("#")[0].strip()
                    result.append([address, match[2], operands, None])
        return result

    def fixed_leaf(self, filename, offset, size):
        ins = self.instructions(filename, offset, size)
        if not ins or len(ins) < 16:
            return None
        stack = []
        for i, (_, op, args, _) in enumerate(ins):
            if op.startswith(("call", "push", "pop", "enter", "leave", "loop")):
                return None
            if _register(args.rsplit(",", 1)[-1]) == _REG_INDEX["rsp"]:
                stack.append(i)
            if op.startswith("xchg") and "%rsp" in args:
                return None
            if op.startswith("j"):
                if not (target := re.match("([0-9a-f]+)", args)):
                    return None
                if not offset <= int(target[1], 16) < offset + size:
                    return None
        if len(stack) != 2:
            return None
        first, last = stack
        if ins[first][1] != "sub" or ins[last][1] != "add":
            return None
        if not (match := re.fullmatch("\\$0x([0-9a-f]+),%rsp", ins[first][2])):
            return None
        frame_size = int(match[1], 16)
        if ins[last][2] != ins[first][2] or first > 2:
            return None
        regs = {"rbx": 3, "rbp": 6, "r12": 12, "r13": 13, "r14": 14, "r15": 15}
        saved = {}
        index = first + 1
        while index < len(ins):
            match = (
                re.fullmatch(
                    "%(rbx|rbp|r12|r13|r14|r15),(0x[0-9a-f]+)?\\(%rsp\\)", ins[index][2]
                )
                if ins[index][1] == "mov"
                else None
            )
            if not match:
                break
            slot = int(match[2] or "0", 16)
            if slot < 0 or slot + 8 > frame_size:
                return None
            saved[regs[match[1]]] = slot
            index += 1
        if len(saved) != 6 or len(set(saved.values())) != 6:
            return None
        # Every accepted body instruction must run after the complete prologue.
        if any(op.startswith("j") for _, op, _, _ in ins[:index]):
            return None
        body_start = ins[index][0]
        restore_start = last - 6
        if restore_start < index:
            return None
        restored = set()
        for row in ins[restore_start:last]:
            match = (
                re.fullmatch(
                    "(0x[0-9a-f]+)?\\(%rsp\\),%(rbx|rbp|r12|r13|r14|r15)", row[2]
                )
                if row[1] == "mov"
                else None
            )
            if not match or saved[regs[match[2]]] != int(match[1] or "0", 16):
                return None
            restored.add(regs[match[2]])
        if len(restored) != 6:
            return None
        if (
            sum(row[1].startswith("ret") for row in ins) != 1
            or ins[-1][1] != "ret"
            or ins[-1][2]
        ):
            return None
        for _, op, args, _ in ins[index:restore_start]:
            if "%rsp" in args.rsplit(",", 1)[-1] and (
                not op.startswith(("cmp", "test"))
            ):
                return None
            if (
                op.startswith("j")
                and int(re.match("([0-9a-f]+)", args)[1], 16) < body_start
            ):
                return None
        return (frame_size, saved, body_start, ins[restore_start][0])

    def call_targets(self, object_name, start, size, symbol_name):
        ins = self.instructions(
            str(self.directory / object_name), start, size, symbol_name
        )
        if not ins:
            return {}
        if any(
            op.startswith("loop") or (op.startswith("jmp") and args.startswith("*"))
            for _, op, args, _ in ins
        ):
            return {}
        indices = {row[0]: i for i, row in enumerate(ins)}
        incoming = {0: (None,) * 16}
        queue = deque([0])
        while queue:
            i = queue.popleft()
            state = list(incoming[i])
            _, op, args, reloc = ins[i]
            dest = re.search(r"(?:^|,)(%\w+)$", args)
            destination = _register(dest[1]) if dest else None
            if op.startswith("call"):
                for r in _CALLER_SAVED:
                    state[r] = None
            elif op.startswith("cmpxchg"):
                state[_REG_INDEX["rax"]] = None
                if destination is not None:
                    state[destination] = None
            elif op in ("cpuid", "rdtsc", "rdtscp", "xgetbv"):
                for name in ("rax", "rbx", "rcx", "rdx"):
                    state[_REG_INDEX[name]] = None
            elif op.startswith(("cmp", "test", "j", "push", "nop")) or op in (
                "ret",
                "vzeroupper",
                "endbr64",
            ):
                pass
            elif op.startswith(("xchg", "xadd")):
                for token in args.split(","):
                    if (r := _register(token)) is not None:
                        state[r] = None
            elif op in ("mul", "imul", "div", "idiv") and "," not in args:
                state[_REG_INDEX["rax"]] = state[_REG_INDEX["rdx"]] = None
            elif op.startswith("leave"):
                state[_REG_INDEX["rbp"]] = state[_REG_INDEX["rsp"]] = None
            elif destination is not None:
                source = args.split(",", 1)[0]
                full_register = dest[1][1:] in _REG_INDEX
                if op == "movabs" and full_register and reloc:
                    state[destination] = reloc
                elif op == "mov" and full_register and source.startswith("%"):
                    r = _REG_INDEX.get(source[1:])
                    state[destination] = state[r] if r is not None else None
                else:
                    state[destination] = None
            elif op.startswith("pop"):
                if (r := _register(args)) is not None:
                    state[r] = None
            elif not op.startswith("v") and not (
                dest and re.fullmatch(r"%[xyz]mm\d+", dest[1])
            ):
                for r in _CALLER_SAVED:
                    state[r] = None
            successors = []
            if op.startswith("j"):
                if target := re.match("([0-9a-f]+)", args):
                    if (j := indices.get(int(target[1], 16))) is not None:
                        successors.append(j)
                if op not in ("jmp", "jmpq") and i + 1 < len(ins):
                    successors.append(i + 1)
            elif not op.startswith(("ret", "ud2")) and i + 1 < len(ins):
                successors.append(i + 1)
            state = tuple(state)
            for j in successors:
                old = incoming.get(j)
                merged = (
                    state
                    if old is None
                    else tuple((a if a == b else None for a, b in zip(old, state)))
                )
                if old != merged:
                    incoming[j] = merged
                    queue.append(j)
        targets = {}
        for i, (_, op, args, reloc) in enumerate(ins[:-1]):
            if not op.startswith("call") or i not in incoming:
                continue
            target = reloc
            if match := re.fullmatch("\\*%(\\w+)", args):
                r = _register(match[1])
                target = incoming[i][r] if r is not None else None
            if target in self.external:
                targets[ins[i + 1][0]] = target
        return targets


def decode(output):
    recording = _Recording(output / "perf.data")
    symbols = sorted(
        json.loads((output / "symbols.json").read_text()),
        key=lambda row: row["address"],
    )
    addresses = [row["address"] for row in symbols]
    external = json.loads((output / "external_calls.json").read_text())

    def lookup(pc):
        index = bisect_right(addresses, pc) - 1
        return (
            symbols[index]
            if index >= 0 and pc < symbols[index]["address"] + symbols[index]["size"]
            else None
        )

    stats = Counter()
    call_sites = {}
    unwinder = _Unwinder(recording.modules())
    analysis = _Analysis(output, external, unwinder)
    try:
        with (
            (output / "samples.txt").open("w") as flat,
            (output / "callchains.txt").open("w") as chains,
        ):
            for ip, weight, registers, stack in recording.samples():
                initial = [registers.get(index, 0) for index in _REGISTERS]
                frames = (
                    unwinder.unwind(initial, registers[7], stack) if registers else []
                )
                if frames and frames[0] != ip:
                    stats["discarded_stacks"] += 1
                    frames = []
                frames = frames or [ip]
                if len(frames) == 1 and registers and lookup(ip) is None:
                    found = unwinder.symbol(ip)
                    if found:
                        start, size, offset = found
                        descriptor = analysis.fixed_leaf(
                            unwinder.path(ip), offset, size
                        )
                        if descriptor:
                            frame_size, saved, body_start, body_end = descriptor
                            if (
                                body_start <= ip - start + offset < body_end
                                and frame_size + 8 <= len(stack)
                            ):
                                ret = struct.unpack_from("<Q", stack, frame_size)[0]
                                if analysis.is_call_return(ret):
                                    resumed = initial.copy()
                                    for register, slot in saved.items():
                                        resumed[register] = struct.unpack_from(
                                            "<Q", stack, slot
                                        )[0]
                                    resumed[7] += frame_size + 8
                                    resumed[16] = ret
                                    callers = unwinder.unwind(
                                        resumed, registers[7], stack
                                    )
                                    if callers:
                                        frames.extend(callers)
                                        stats["fixed_frame_samples"] += 1
                resolved = []
                for index, pc in enumerate(frames):
                    symbol = lookup(pc - bool(index))
                    if symbol:
                        name, library = symbol["name"], "[jit]"
                        if index:
                            target = analysis.call_targets(
                                symbol["object"], symbol["offset"], symbol["size"], name
                            ).get(pc - symbol["address"] + symbol["offset"])
                            if target:
                                call_sites[pc] = target
                    else:
                        name, library = (
                            unwinder.name(pc - bool(index)),
                            unwinder.path(pc - bool(index)),
                        )
                    resolved.append((pc, name, library))
                flat.write(f"{weight} {ip:x} {resolved[0][1]} ({resolved[0][2]})\n")
                chains.write(f"{weight}\n")
                for pc, name, library in resolved:
                    chains.write(f"\t{pc:x} {name} ({library})\n")
                chains.write("\n")
    finally:
        analysis.close()
        unwinder.close()
    (output / "call_sites.json").write_text(json.dumps(call_sites, indent=2) + "\n")
    stats["lost_samples"] = recording.lost
    metadata = dict(stats)
    if analysis.errors:
        metadata["assembly_errors"] = analysis.errors
    (output / "recovery.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
