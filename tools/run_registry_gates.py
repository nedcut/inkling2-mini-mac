"""Run local-ai-registry's six recipe gates (load, chat, reasoning, tools, context, speed)
against a running ./serve.sh, using the registry's own gate code.

  curl -fsSL -o /tmp/lab.py \
    https://raw.githubusercontent.com/sybil-solutions/local-ai-registry/<commit>/lab/lab.py
  python3 -I tools/run_registry_gates.py --lab /tmp/lab.py --ctx 32768 [--endpoint URL] [--out FILE]

The context gate sends one prompt that fills 85% of --ctx and checks that a code
planted near its end comes back, so --ctx must be what this server can hold
(serve.sh's MAX_CONTEXT). Writes the full evidence as JSON.
"""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--lab", required=True, help="path to local-ai-registry's lab/lab.py")
ap.add_argument("--endpoint", default="http://127.0.0.1:8790")
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--out")
a = ap.parse_args()

lab_path = pathlib.Path(a.lab).resolve()
spec = importlib.util.spec_from_file_location("registry_lab", lab_path)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

launch = {"ctx": a.ctx, "seqs": 1, "command": ["./serve.sh"], "env": {}}
t0 = time.time()
passed, proof, evidence = lab.gates(a.endpoint.rstrip("/"), launch)
result = {
    "passed": passed,
    "proof": proof,
    "evidence": evidence,
    "ctx": a.ctx,
    "lab_sha256": hashlib.sha256(lab_path.read_bytes()).hexdigest(),
    "seconds": round(time.time() - t0, 1),
}
text = json.dumps(result, indent=2, default=str)
if a.out:
    pathlib.Path(a.out).write_text(text)
print(json.dumps({"passed": passed, "proof": proof,
                  "context": evidence.get("context"), "speed": evidence.get("speed")}, default=str))
sys.exit(0 if passed else 1)
