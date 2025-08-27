from dataclasses import dataclass, field
from typing import Any, Union

from pytensor.graph import Op
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.unify import OpInstance


@dataclass(frozen=True, eq=False)
class MatchPattern:
    name: str | None
    pattern: tuple
    _var_to_standard: dict[str, int] = field(default_factory=dict)
    _standard_to_var: dict[int, str] = field(default_factory=dict)

    def __repr__(self):
        if self.name is not None:
            return self.name
        return str(self.pattern)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def get_standard_var_name(self, var: str) -> str:
        """Get a canonicalized variable name for a pattern variable.
        This increases sharing of paths in the trie as only uniqueness of names matters for the purpose of unification.
        """
        standard_var = self._var_to_standard.get(var, None)
        if standard_var is None:
            standard_var = f"{len(self._var_to_standard)}"
            self._var_to_standard[var] = standard_var
            self._standard_to_var[standard_var] = var
        return standard_var


@dataclass(frozen=True)
class Literal:
    # Wrapper class to signal that a pattern is a literal value, not a pattern variable
    pattern: Any


@dataclass(frozen=True)
class Asterisk:
    # Wrapper class to signal that a pattern is a wildcard matching zero or more variables
    variable: str


@dataclass(frozen=True)
class TrieNode:
    # Class for Op level trie nodes
    # Each node has edges for exact Op matches, Op type matches, variable matches, and
    # edges for starting parametrized Op matches (which lead to ParameterTrieNode)
    # Terminal patterns are stored at the nodes where patterns end
    op_edges: dict[Op, "TrieNode"] = field(default_factory=dict)
    op_type_edges: dict[type[Op], "TrieNode"] = field(default_factory=dict)
    start_parameter_edges: dict[type[Op], "ParameterTrieNode"] = field(
        default_factory=dict
    )
    variable_edges: dict[str, "TrieNode"] = field(default_factory=dict)
    asterisk_edges: dict[str, "TrieNode"] = field(
        default_factory=dict
    )  # New: asterisk variable edges
    terminal_patterns: list[MatchPattern] = field(default_factory=list)


@dataclass(frozen=False)
class ParameterTrieNode:
    # Class for Op parameter level trie nodes
    # Each node has edges for matching Op parameters (key, pattern) pairs
    # (where pattern can be a variable name, an Op type, a literal value, or a nested parametrized Op (OpType, {param: value, ...}))

    # A ParameterTrieNode may have multiple parameter edges to move to the next ParameterTrieNode
    # A ParameterTrieNode may have an end_parameter_edge, to move back to the outer TrieNode/ ParameterTrieNode
    # This allows different patterns to match a different number of parameters.
    # Parameters are arranged in alphabetical order to help sharing of common paths.

    # A ParameterTrieNode may also have a sub_op_parameter_edge, to start matching parameters of a nested parametrized Op
    # A sub_op_parameter_edge always follows a parameter_edge for the same parameter key and op type.

    parameter_edges: dict[tuple[str, Any], "ParameterTrieNode"] = field(
        default_factory=dict
    )
    sub_op_parameter_edge: tuple[str, "ParameterTrieNode"] | None = field(default=None)

    # A ParameterTrieNode may end up followed by a ParameterTrieNode, if it was a nested parametrized op
    # Or with a regular TrieNode, if it was the end of a parametrized op pattern
    end_parameter_edge: Union["TrieNode", "ParameterTrieNode"] | None = field(
        default=None
    )


@dataclass(frozen=False)
class Trie:
    root_node: TrieNode = field(default_factory=TrieNode)
    op_type_match_cache: dict = field(default_factory=dict)

    def add_pattern(self, pattern: MatchPattern | tuple):
        """Expand Trie with new pattern"""
        self.op_type_match_cache.clear()

        if not isinstance(pattern, MatchPattern):
            pattern = MatchPattern(None, pattern)

        def get_keyed_edge(edges_dict, key, trie_class=TrieNode):
            next_trie_node = edges_dict.get(key, None)
            if next_trie_node is None:
                edges_dict[key] = next_trie_node = trie_class()
            return next_trie_node

        def recursive_insert_params(trie_node, parameters, nested=False):
            assert isinstance(trie_node, ParameterTrieNode)
            if not parameters:
                # Base case: We consumed all the parameters. Add an end_parameter edge to signal we're done
                if trie_node.end_parameter_edge is None:
                    trie_node.end_parameter_edge = (
                        ParameterTrieNode() if nested else TrieNode()
                    )
                return trie_node.end_parameter_edge

            (item_key, item_pattern), *rest_key_pattern_pairs = parameters

            if isinstance(item_pattern, OpInstance):
                # Nested parametrized op
                sub_op_type, sub_parameters = (
                    item_pattern.op_type,
                    item_pattern.parameters,
                )
                # Start with a parameter edge for the op parameter
                start_trie_node = get_keyed_edge(
                    trie_node.parameter_edges,
                    (item_key, sub_op_type),
                    trie_class=ParameterTrieNode,
                )
                if item_pattern.parameters:
                    # Add a sub_op_parameter edge to start matching the nested Op parameters
                    # A trie node can only have one sub_op_parameter edge, since it's always preceded by a parameter edge
                    if start_trie_node.sub_op_parameter_edge is None:
                        start_trie_node.sub_op_parameter_edge = (
                            item_key,
                            ParameterTrieNode(),
                        )
                    (sub_op_key, sub_op_trie_node) = (
                        start_trie_node.sub_op_parameter_edge
                    )
                    assert sub_op_key == item_key
                    next_trie_node = recursive_insert_params(
                        sub_op_trie_node, sub_parameters, nested=True
                    )
                else:
                    # No parameters, so we can directly move to the next trie node
                    next_trie_node = start_trie_node
            else:
                # Simple parameter pattern: add a parameter edge
                if isinstance(item_pattern, str):
                    # Pattern variable, replace with a unique variable name
                    item_pattern = pattern.get_standard_var_name(item_pattern)
                # All edges (including variables) go through parameter_edges
                # TODO: Consider splitting variable edges into a separate dict for faster matching
                next_trie_node = get_keyed_edge(
                    trie_node.parameter_edges,
                    (item_key, item_pattern),
                    trie_class=ParameterTrieNode,
                )
            # Recurse with the rest of the parameters
            return recursive_insert_params(
                next_trie_node, rest_key_pattern_pairs, nested=nested
            )

        def recursinve_insert(trie_node, sub_pattern):
            if not sub_pattern:
                # Base case: we've consumed the entire pattern
                trie_node.terminal_patterns.append(pattern)
                return

            head, *tail = sub_pattern
            if isinstance(head, tuple):
                # ((op, input1, input2, ...), ...)
                head_head, *head_tail = head
                return recursinve_insert(trie_node, (head_head, *head_tail, *tail))

            if isinstance(head, OpInstance) and head.parameters:
                op_type, parameters = head.op_type, head.parameters
                # Start with an edge for the op type
                next_trie_node = get_keyed_edge(
                    trie_node.start_parameter_edges,
                    op_type,
                    trie_class=ParameterTrieNode,
                )
                # Recurse into the parameters, with parameter edges
                next_trie_node = recursive_insert_params(next_trie_node, parameters)
            else:
                key = head
                if isinstance(head, Op):
                    edge_type = trie_node.op_edges
                elif isinstance(head, type) and issubclass(head, Op):
                    edge_type = trie_node.op_type_edges
                elif isinstance(head, OpInstance):
                    # Empty ParametrizedOp, handle with a simple op_type edge
                    assert not head.parameters
                    key = head.op_type
                    edge_type = trie_node.op_type_edges
                elif isinstance(head, str):
                    key = pattern.get_standard_var_name(head)
                    edge_type = trie_node.variable_edges
                elif isinstance(head, Asterisk):
                    key = pattern.get_standard_var_name(head.variable)
                    edge_type = trie_node.asterisk_edges
                else:
                    raise TypeError(f"Invalid head type {type(head)}: {head}")
                next_trie_node = get_keyed_edge(edge_type, key)

            # Recurse with the tail of the pattern
            recursinve_insert(next_trie_node, tail)

        recursinve_insert(self.root_node, pattern.pattern)

    def match(self, variable):
        if not isinstance(variable, Variable):
            return False

        def find_op_type_edge_matches(edges_dict, op: Op):
            type_op = type(op)
            cache_key = (id(edges_dict), type_op)
            if cache_key in self.op_type_match_cache:
                yield from self.op_type_match_cache[cache_key]
                return

            self.op_type_match_cache[cache_key] = matches = [
                match
                for base_cls in type_op.mro()
                if (match := edges_dict.get(base_cls)) is not None
            ]
            yield from matches

        def find_op_matches(trie_node: TrieNode, op: Op):
            if (next_trie_node := trie_node.op_edges.get(op)) is not None:
                yield next_trie_node

            yield from find_op_type_edge_matches(trie_node.op_type_edges, op)

        def recursive_match(
            trie_node: TrieNode | ParameterTrieNode,
            subject_pattern: tuple[Variable, tuple[Variable, ...]],
            subs: dict[str, Any],
            num_op_inputs: tuple,
        ):
            if isinstance(trie_node, TrieNode):
                # Base case, terminal patterns are successfully matched
                # whenever trie node is reached with no subject pattern left to unify
                if not subject_pattern:
                    for terminal_pattern in trie_node.terminal_patterns:
                        # Convert the canonicalized variable names back to the original pattern variable names
                        d = terminal_pattern._standard_to_var
                        yield terminal_pattern, {d[k]: v for k, v in subs.items()}

                # Unify asterisk variables
                # This must be the last pattern for the current op's inputs
                for asterisk_var, next_trie_node in trie_node.asterisk_edges.items():
                    remaining_n_inputs, tail_n_inputs = num_op_inputs
                    consumed_vars = subject_pattern[:remaining_n_inputs]
                    remaining_subject = subject_pattern[remaining_n_inputs:]
                    subs_copy = subs

                    if asterisk_var in subs:
                        if subs[asterisk_var] != consumed_vars:
                            continue  # mismatch
                    else:
                        subs_copy = subs.copy()
                        subs_copy[asterisk_var] = consumed_vars
                    yield from recursive_match(
                        next_trie_node,
                        remaining_subject,
                        subs_copy,
                        num_op_inputs=tail_n_inputs,
                    )

                if not subject_pattern:
                    # Nothing left to match
                    return None

                head, *tail = subject_pattern
                assert isinstance(head, Variable), (type(head), head)

                # Unify variable patterns
                for (
                    variable_pattern,
                    next_trie_node,
                ) in trie_node.variable_edges.items():
                    subs_copy = subs
                    if variable_pattern in subs:
                        if subs[variable_pattern] != head:
                            continue  # mismatch
                    else:
                        subs_copy = subs.copy()
                        subs_copy[variable_pattern] = head

                    remaining_n_inputs, tail_n_inputs = num_op_inputs
                    if remaining_n_inputs == 0:
                        # We've exhausted the inputs for the current op, this next variable belongs to the next input of the outer Op
                        remaining_n_inputs, tail_n_inputs = tail_n_inputs
                    assert (
                        remaining_n_inputs > 0
                    ), "Number of inputs to consume is smaller than expected. Perhaps missing an Asterisk pattern?"
                    yield from recursive_match(
                        next_trie_node,
                        tail,
                        subs_copy,
                        (remaining_n_inputs - 1, tail_n_inputs),
                    )

                if head.owner is None:
                    # head is a root variable, can only be matched to wildcard patterns above
                    return False
                head_op = head.owner.op

                # Match exact op or type op (including subclasses)
                # We consume the head variable and extend the tail pattern with its inputs
                for next_trie_node in find_op_matches(trie_node, head_op):
                    yield from recursive_match(
                        next_trie_node,
                        (*head.owner.inputs, *tail),
                        subs,
                        (len(head.owner.inputs), num_op_inputs),
                    )

                # Match start of parametrized op pattern
                for next_trie_node in find_op_type_edge_matches(
                    trie_node.start_parameter_edges, head_op
                ):
                    # We place the Op variable at the head of the subject pattern
                    # And extend the tail pattern with the inputs of the head variable, just like a regular op match
                    yield from recursive_match(
                        next_trie_node,
                        (head_op, *head.owner.inputs, *tail),
                        subs,
                        (len(head.owner.inputs), num_op_inputs),
                    )

            else:  # ParameterTrieNode
                head_op, *tail = subject_pattern
                assert isinstance(head_op, Op), (type(head_op), head_op)

                # Exit parametrized op pattern matching
                if (next_trie_node := trie_node.end_parameter_edge) is not None:
                    # We discard the head variable and keep working on the tail pattern
                    yield from recursive_match(
                        next_trie_node, tail, subs, num_op_inputs
                    )

                # Match op parameters
                for (
                    op_param_key,
                    op_param_pattern,
                ), next_trie_node in trie_node.parameter_edges.items():
                    op_param_value = getattr(head_op, op_param_key)
                    subs_copy = subs

                    # Match variable pattern
                    if isinstance(op_param_pattern, str):
                        if op_param_pattern in subs:
                            if subs[op_param_pattern] != op_param_value:
                                continue  # mismatch
                        else:
                            subs_copy = subs.copy()
                            subs_copy[op_param_pattern] = op_param_value
                    # Match op type
                    elif isinstance(op_param_pattern, type) and issubclass(
                        op_param_pattern, Op
                    ):
                        if not isinstance(op_param_value, op_param_pattern):
                            continue  # mismatch
                    # Match literal value
                    elif isinstance(op_param_pattern, Literal):
                        if op_param_value != op_param_pattern.pattern:
                            continue  # mismatch
                    # Match exact value
                    elif op_param_value != op_param_pattern:
                        continue  # mismatch

                    # We arrive here if there was no mismatch
                    # For parameter edges, we continue to the next trie_node with the same pattern
                    # as we may still need to check other parameters from the same Op
                    # We'll eventually move to the tail pattern via an end_parameter edge
                    yield from recursive_match(
                        next_trie_node, subject_pattern, subs_copy, num_op_inputs
                    )

                # Match nested op parametrizations
                # This always follows an op parameter edge
                if trie_node.sub_op_parameter_edge is not None:
                    (sub_op_param_key, next_trie_node) = trie_node.sub_op_parameter_edge
                    sub_op = getattr(head_op, sub_op_param_key)
                    # For sub_op parameter edges, we continue to the next trie_node with the sub_op as the head
                    yield from recursive_match(
                        next_trie_node, (sub_op, *subject_pattern), subs, num_op_inputs
                    )
            return None

        yield from recursive_match(self.root_node, (variable,), {}, ())
        return None
