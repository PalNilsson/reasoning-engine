#!/usr/bin/env python3
"""
Utility script for working with Ask PanDA JSONL training data.

Current behavior:
  * Reads a JSONL file where each line is a JSON object with a `messages` list.
  * Extracts and prints all user prompts (role == "user").

Future behavior (pseudo-code included but commented out):
  * For each conversation:
      - Extract the user prompt.
      - Send it to Ask PanDA (via HTTP, CLI, or SDK).
      - Insert Ask PanDA's response into the final assistant message.
      - Write updated conversations back to a JSONL file.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and (later) fill Ask PanDA JSONL training data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL file (one conversation per line).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path to output JSONL file with updated conversations. "
            "If omitted, the script will NOT write anything and only print prompts."
        ),
    )
    return parser.parse_args(argv)


def extract_user_prompts(messages: List[Dict[str, Any]]) -> List[str]:
    """Return a list of user prompts from a single conversation."""
    prompts: List[str] = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                prompts.append(content)
    return prompts


def find_final_assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the final assistant message (channel == 'final'), if present."""
    # Convention: the final assistant message is usually the last with role=assistant, channel=final
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("channel") == "final":
            return msg
    return None


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    try:
        infile = open(args.input, "r", encoding="utf-8")
    except OSError as exc:
        print(f"Error opening input file '{args.input}': {exc}", file=sys.stderr)
        sys.exit(1)

    outfile = None
    if args.output:
        try:
            outfile = open(args.output, "w", encoding="utf-8")
        except OSError as exc:
            print(f"Error opening output file '{args.output}': {exc}", file=sys.stderr)
            infile.close()
            sys.exit(1)

    try:
        for line_idx, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                conv: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[Line {line_idx}] JSON decode error: {exc}",
                    file=sys.stderr,
                )
                continue

            messages = conv.get("messages")
            if not isinstance(messages, list):
                print(
                    f"[Line {line_idx}] No 'messages' list found, skipping.",
                    file=sys.stderr,
                )
                continue

            # --- 1) Extract and print all user prompts ---
            user_prompts = extract_user_prompts(messages)
            print(f"Conversation #{line_idx}:")
            for i, prompt in enumerate(user_prompts, start=1):
                print(f"  [user #{i}] {prompt}")
            print()

            # --- 2) Pseudo-code: Call Ask PanDA and update the final assistant answer ---
            # This is where you would:
            #   - Take the main user prompt (e.g. the last one)
            #   - Call Ask PanDA
            #   - Insert its answer into the final assistant message
            #
            # Example skeleton:
            #
            # if user_prompts:
            #     main_prompt = user_prompts[-1]
            #
            #     # PSEUDO-CODE: call Ask PanDA (replace with real API call/CLI)
            #     # ask_panda_answer = call_ask_panda(main_prompt)
            #
            #     # For now, we just fake an answer or leave it unchanged:
            #     ask_panda_answer = "DUMMY ANSWER from Ask PanDA for prompt: " + main_prompt
            #
            #     final_msg = find_final_assistant_message(messages)
            #     if final_msg is not None:
            #         # Only overwrite if it's empty or placeholder
            #         current_content = final_msg.get("content", "")
            #         if not current_content:
            #             final_msg["content"] = ask_panda_answer
            #     else:
            #         # If there is no final assistant message, we can append one
            #         messages.append(
            #             {
            #                 "role": "assistant",
            #                 "channel": "final",
            #                 "content": ask_panda_answer,
            #             }
            #         )

            # --- 3) Optional: write updated conversation back out as JSONL ---
            if outfile is not None:
                json.dump(conv, outfile, ensure_ascii=False)
                outfile.write("\n")

    finally:
        infile.close()
        if outfile is not None:
            outfile.close()


if __name__ == "__main__":
    main()
