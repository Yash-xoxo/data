#!/bin/bash

# ================= CONFIG =================
OLLAMA_URL="http://localhost:11434/api/generate"

MODEL_CURIOUS="granite4:350m"
MODEL_EXPLAINER="gamma3:4b"
MODEL_SUMMARY="granite4:350m"

CTX=512
PREDICT=128
ROUNDS=4   # number of Q/A cycles

CHAT_FILE="chat.log"
SUMMARY_FILE="summary.txt"
TMP_FILE="chat.tmp"

# =========================================

read -p "Enter the topic: " TOPIC
echo ""

# Clean previous files
rm -f "$CHAT_FILE" "$SUMMARY_FILE" "$TMP_FILE"
touch "$CHAT_FILE" "$TMP_FILE"

echo "Topic: $TOPIC" >> "$CHAT_FILE"
echo "----------------------------------------" >> "$CHAT_FILE"

ollama_call () {
  local MODEL="$1"
  local PROMPT="$2"

  curl -s "$OLLAMA_URL" -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$PROMPT\",
    \"stream\": false,
    \"options\": {
      \"num_ctx\": $CTX,
      \"num_predict\": $PREDICT,
      \"temperature\": 0.7
    }
  }" | sed -n 's/.*"response":"\([^"]*\)".*/\1/p'
}

# ---------- Conversation Loop ----------
for ((i=1; i<=ROUNDS; i++)); do

  # ---- Model 1: Curious Questioner ----
  PROMPT_Q="You are a very curious person.
Topic: $TOPIC

Conversation so far:
$(cat "$TMP_FILE")

Ask ONE thoughtful, curiosity-driven question.
Stay human. Avoid summaries."

  QUESTION=$(ollama_call "$MODEL_CURIOUS" "$PROMPT_Q")

  echo "[Curious Model]" >> "$CHAT_FILE"
  echo "$QUESTION" >> "$CHAT_FILE"
  echo "" >> "$CHAT_FILE"

  echo "Curious: $QUESTION" >> "$TMP_FILE"

  # ---- Model 2: Detailed Explainer ----
  PROMPT_A="You are an expert explainer.

Topic: $TOPIC

Question:
$QUESTION

Give a clear, deep, detailed explanation.
Be factual and structured."

  ANSWER=$(ollama_call "$MODEL_EXPLAINER" "$PROMPT_A")

  echo "[Explainer Model]" >> "$CHAT_FILE"
  echo "$ANSWER" >> "$CHAT_FILE"
  echo "" >> "$CHAT_FILE"

  echo "Explainer: $ANSWER" >> "$TMP_FILE"

  # Small pause to avoid overload
  sleep 1
done

# ---------- Summarization ----------
SUMMARY_PROMPT="You are a thoughtful human reader.

Below is a conversation where one person asked curious questions
and another explained in detail.

Your task:
- Read everything
- Extract key ideas
- Write a natural, human-like conclusion
- Do NOT mention AI or models

Conversation:
----------------
$(cat "$CHAT_FILE")
----------------

Final conclusion:"

SUMMARY=$(ollama_call "$MODEL_SUMMARY" "$SUMMARY_PROMPT")

echo "$SUMMARY" > "$SUMMARY_FILE"

# ---------- Cleanup ----------
rm -f "$TMP_FILE"

echo ""
echo "========================================"
echo "Conversation saved to: $CHAT_FILE"
echo "Summary saved to:      $SUMMARY_FILE"
echo "Done."
echo "========================================"
