# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


def get_and_increment_counter():
    with open("counter.txt", "r") as f:
        counter = int(f.read().strip())
    counter += 1
    with open("counter.txt", "w") as f:
        f.write(str(counter))
    return counter - 1


def get_last_counter():
    with open("counter.txt", "r") as f:
        counter = int(f.read().strip())
    return counter - 1
