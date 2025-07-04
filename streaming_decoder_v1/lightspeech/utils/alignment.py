from typing import List
from dataclasses import dataclass

import torch


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self):
        return self.end - self.start


def get_trellis(
    emission: torch.Tensor,
    tokens: List[int],
    blank: int,
) -> torch.Tensor:

    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )

    return trellis


def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: List[int],
    blank: int,
) -> List[Point]:

    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0]
        prob = prob.exp().item()

        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")

    return path[::-1]


def merge_tokens(
    path: List[Point],
    tokens: List[int],
    feature_length: int,
    audio_length: int,
) -> List[Segment]:

    i1, i2 = 0, 0
    segments = []

    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1

        label = tokens[path[i1].token_index]
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)

        start = path[i1].time_index / feature_length * audio_length
        end = (path[i2 - 1].time_index + 1) / feature_length * audio_length

        segments.append(Segment(label, start, end, score))

        i1 = i2

    return segments


def merge_words(segments: List[Segment], silence: str) -> List[Segment]:

    words = []
    i1, i2 = 0, 0

    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == silence:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )

                words.append(
                    Segment(
                        word,
                        segments[i1].start,
                        segments[i2 - 1].end,
                        score,
                    )
                )

            i1 = i2 + 1
            i2 = i1

        else:
            i2 += 1

    return words
