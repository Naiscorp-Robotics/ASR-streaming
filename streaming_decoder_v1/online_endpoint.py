from dataclasses import dataclass


@dataclass
class OnlineEndpointRule:
    # If True, for this endpointing rule to apply there must
    # be nonsilence in the best-path traceback.
    must_contain_nonsilence: bool

    # This endpointing rule requires duration of trailing silence
    # (in seconds) to be >= this value.
    min_trailing_silence: float

    # This endpointing rule requires utterance-length (in seconds)
    # to be >= this value.
    min_utterance_length: float

    # This endpointing rule the log propablity of final state *(-5) <= this value.
    # such as the probality is 0.1, log is -1 and the value is 5
    max_relative_cost: float


def load_endpointing_rule(args: dict):
    """
    Args:
        args:
        It contains the arguments parsed from
        :func:`add_online_endpoint_arguments`
    """
    rules = {}
    for rule_name in args.keys():
        rule = OnlineEndpointRule(
            must_contain_nonsilence = args[rule_name]['must_contain_nonsilence'],
            min_trailing_silence    = args[rule_name]['min_trailing_silence'],
            min_utterance_length    = args[rule_name]['min_utterance_length'],
            max_relative_cost       = args[rule_name]['max_relative_cost']
            )
        rules[rule_name] = rule
    return rules


def _rule_activated(
    rule: OnlineEndpointRule,
    trailing_silence: float,
    utterance_length: float,
    relative_cost: float
):
    """
    Args:
        rule:
            The rule to be checked.
        trailing_silence:
            Trailing silence in seconds.
        utterance_length:
            Number of frames in seconds decoded so far.
    Returns:
        Return True if the given rule is activated; return False otherwise.
    """
    contains_nonsilence = utterance_length > trailing_silence
    is_activate = (
            (contains_nonsilence or not rule.must_contain_nonsilence)
            and (trailing_silence   >=  rule.min_trailing_silence)
            and (relative_cost      <   rule.max_relative_cost)
            and (utterance_length   >=  rule.min_utterance_length)
    )
    return is_activate


def detect_endpointing(
    rule: dict,
    utterance_length: float,
    trailing_silence: float,
    relative_cost: float,
) -> bool:
    """
    Args:
        config:
            The endpoint config to be checked.
        utterance_length:
            Durations of decoded utterance decoded so far.
        trailing_silence:
            Duration of trailing silence.
        relative_cost:
            Score of transcripts to determize endpointing by linguistic information.
    Returns:
        Return True if any rule in `config` is activated; return False otherwise.
    """

    for rule_name in rule.keys():
        if _rule_activated(rule[rule_name], trailing_silence, utterance_length, relative_cost):
            return True, rule_name, \
                trailing_silence - rule[rule_name].min_trailing_silence

    return False, None, None
