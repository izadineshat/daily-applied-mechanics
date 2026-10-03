import os
import datetime
from openai import OpenAI

# دریافت توکن OpenRouter از Secrets گیت‌هاب
api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

today_str = datetime.datetime.now().strftime("%Y-%m-%d")

prompt = """
Write a standalone, production-ready, well-documented Python script for Mechanical Engineering (Applied Mechanics / Solid Mechanics / Applied Design).
Topics include:
- Finite Element Analysis (1D/2D truss/beam/plate stiffness matrices)
- Stress & Strain Transformations (Mohr's circle, Von Mises, Tresca)
- Mechanical Vibrations (modal analysis, SDOF/MDOF damped response, state-space)
- Fatigue & Fracture Mechanics (S-N curves, Goodman/Gerber diagrams, Paris' Law)
- Structural Optimization (beam cross-section weight minimization, GA/gradient methods)
- Rotordynamics / Contact Mechanics / Failure Theories

Requirements:
1. Include clean object-oriented or modular functions.
2. Use standard libraries (numpy, scipy, matplotlib).
3. Include realistic engineering numerical defaults and a working `if __name__ == '__main__':` block.
4. Output ONLY valid executable Python code without markdown triple-backtick fences or introductory text.
"""

# تنظیم مدل روی روت رایگان اوپن‌روتر
response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "system",
            "content": "You are an expert computational mechanical engineer and Python developer. Return only pure Python code without any explanation.",
        },
        {"role": "user", "content": prompt},
    ],
)

code_content = response.choices[0].message.content.strip()

# پاک‌سازی تگ‌های احتمالی Markdown
if code_content.startswith("```python"):
    code_content = code_content.removeprefix("```python").removesuffix("```").strip()
elif code_content.startswith("```"):
    code_content = code_content.removeprefix("```").removesuffix("```").strip()

os.makedirs("scripts", exist_ok=True)
filename = f"scripts/mech_design_{today_str}.py"

with open(filename, "w", encoding="utf-8") as f:
    f.write(code_content)

print(f"Successfully generated: {filename}")
