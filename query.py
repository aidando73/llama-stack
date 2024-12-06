# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient

bank_id = "bank_pdf_paper"

client = LlamaStackClient(
    base_url="http://localhost:5001",
)


def query2context_pdf(query, bank_id):
    response = client.memory.query(
        bank_id=bank_id,
        query=[query],
    )

    context = ""
    for chunk, score in zip(response.chunks, response.scores):
        context += str(chunk) + str(score) + "\n"  # chunk.content
    print(context)
    return context


query = "How big is the context window?"
query2context_pdf(query, bank_id)
