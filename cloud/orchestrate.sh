#!/usr/bin/env bash
# Detached orchestrator: wait for env setup + the 4 LoRAs, then run the eval matrix.
# Survives SSH disconnects (launched via setsid). Progress -> /workspace/orch.log
cd /workspace
echo "[orch] waiting for SETUP DONE..."
for i in $(seq 1 90); do grep -q "SETUP DONE" setup.log 2>/dev/null && break; sleep 20; done
grep -q "SETUP DONE" setup.log 2>/dev/null && echo "[orch] setup done" || { echo "[orch] setup TIMEOUT"; tail -5 setup.log; }

echo "[orch] waiting for 4 LoRAs..."
for i in $(seq 1 60); do
  n=$(ls lora150/lora_weights.safetensors lora250/lora_weights.safetensors lora350/lora_weights.safetensors lora500/lora_weights.safetensors 2>/dev/null | wc -l)
  [ "$n" -ge 4 ] && break; sleep 15
done
echo "[orch] LoRAs present: $(ls lora*/lora_weights.safetensors 2>/dev/null | wc -l)/4"

echo "[orch] running eval matrix..."
source venv/bin/activate
python infer_eval.py
echo "[orch] ORCHESTRATE DONE"
