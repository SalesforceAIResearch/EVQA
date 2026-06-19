# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import re


def parse_question(question):
    return re.sub(r'\s+', ' ', question).strip()
