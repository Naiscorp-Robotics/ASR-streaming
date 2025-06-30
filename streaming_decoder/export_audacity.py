import sys

from utils import AudacitySegment, convert2audacity

f = open(sys.argv[1], "r")

data = []
start = 0.0
for i, line in enumerate(f.readlines()):
    if i == 0:
        output = line.strip().split(".wav")[0].split("/")[-1]+".txt"
    if "segment:" in line:
        text = ""
        data_point = AudacitySegment()
        data_point.start_time = start
    if "text:" in line:
        text = line.strip().split(":")[-1]
        data_point.label = text
    if "segment_length" in line:
        start += float(line.strip().split(" ")[-1].strip())
        data_point.stop_time = start
        data.append(data_point)
convert2audacity(data, output)