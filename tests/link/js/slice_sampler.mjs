// Experimental consumer of a PyTensor JSProgram. The sampling loop never calls Python.
import fs from 'node:fs';
import vm from 'node:vm';

const {program, initial, draws, tune, width, maxSteps, seed} = JSON.parse(
  fs.readFileSync(0, 'utf8'),
);
if (program.inputs.length !== 1 || program.inputs[0].ndim !== 0 ||
    program.outputs.length !== 1 || program.outputs[0].ndim !== 0) {
  throw Error('slice sampler expects one scalar input and one scalar logp output');
}
if (!Number.isFinite(initial) || !Number.isInteger(draws) || draws < 1 ||
    !Number.isInteger(tune) || tune < 0 || !(width > 0) ||
    !Number.isInteger(maxSteps) || maxSteps < 1 || !Number.isInteger(seed)) {
  throw Error('invalid slice-sampling settings');
}

vm.runInThisContext(program.source);
function strides(shape) {
  const result = Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    result[i] = stride;
    stride *= shape[i];
  }
  return result;
}
for (const [index, constant] of program.constants.entries()) {
  const bytes = Buffer.from(constant.data, 'base64');
  const values = new Float64Array(bytes.byteLength / 8);
  Buffer.from(values.buffer).set(bytes);
  __ptjs.constants[index] = {d: values, s: constant.shape, t: strides(constant.shape)};
}
__ptjs.setInputShape(0, []);

let state = seed >>> 0;
if (state === 0) state = 1;
function uniform() {
  state ^= state << 13;
  state ^= state >>> 17;
  state ^= state << 5;
  return ((state >>> 0) + 0.5) / 4294967296;
}

let evaluations = 0;
function logp(x) {
  __ptjs.inputs[0].d[0] = x;
  __ptjs.execute();
  evaluations++;
  const value = __ptjs.outputs()[0].d[0];
  if (Number.isNaN(value)) throw Error(`logp is NaN at ${x}`);
  return value;
}

let x = initial;
let current = logp(x);
if (!Number.isFinite(current)) throw Error('initial point has non-finite logp');
const samples = new Array(draws);
const start = process.hrtime.bigint();
for (let iteration = 0; iteration < tune + draws; iteration++) {
  const level = current + Math.log(uniform());
  let left = x - width * uniform();
  let right = left + width;
  let leftSteps = Math.floor(maxSteps * uniform());
  let rightSteps = maxSteps - 1 - leftSteps;
  while (leftSteps-- > 0 && logp(left) > level) left -= width;
  while (rightSteps-- > 0 && logp(right) > level) right += width;

  let accepted = false;
  for (let attempt = 0; attempt < 10000; attempt++) {
    const candidate = left + uniform() * (right - left);
    const candidateLogp = logp(candidate);
    if (candidateLogp >= level) {
      x = candidate;
      current = candidateLogp;
      accepted = true;
      break;
    }
    if (candidate < x) left = candidate;
    else right = candidate;
  }
  if (!accepted) throw Error('slice sampler failed to shrink bracket');
  if (iteration >= tune) samples[iteration - tune] = x;
}
const sampleSeconds = Number(process.hrtime.bigint() - start) / 1e9;
process.stdout.write(JSON.stringify({samples, evaluations, sampleSeconds}));
