# file: sapnote_2431747_service.py  (part 1 of 2)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, re, json

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2431747 FI/CO Obsolete Table Assessment & Remediation")

# ===== Table Replacement Maps =====
TABLE_MAPPING = {
    "BSIS": {"source": "BSEG", "view": True},
    "BSAS": {"source": "BSEG", "view": True},
    "BSIK": {"source": "BSEG", "view": True},
    "BSAK": {"source": "BSEG", "view": True},
    "BSID": {"source": "BSEG", "view": True},
    "BSAD": {"source": "BSEG", "view": True},
    "GLT0": {"source": "ACDOCA/BSEG", "view": True},
    "COEP": {"source": "ACDOCA", "view": True},
    "COSP": {"source": "ACDOCA", "view": True},
    "COSS": {"source": "ACDOCA", "view": True},
    "MLIT": {"source": "ACDOCA", "view": True},
    "ANEP": {"source": "ACDOCA", "view": True},
    "ANLP": {"source": "ACDOCA", "view": True},
}
NO_VIEW_TABLES = {"FAGLFLEXA", "FAGLFLEXT"}

# ===== Regex patterns =====
SELECT_RE = re.compile(
    r"""(?P<full>
            SELECT\s+(?:SINGLE\s+)?        
            (?P<fields>[\w\s,*]+)          
            \s+FROM\s+(?P<table>\w+)       
            (?P<middle>.*?)                
            (?:
                (?:INTO\s+TABLE\s+(?P<into_tab>[\w@()\->]+))
              | (?:INTO\s+(?P<into_wa>[\w@()\->]+))
            )
            (?P<tail>.*?)
        )\.""",
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
UPDATE_RE = re.compile(r"(UPDATE\s+\w+[\s\S]*?\.)", re.IGNORECASE)
DELETE_RE = re.compile(r"(DELETE\s+FROM\s+\w+[\s\S]*?\.)", re.IGNORECASE)
INSERT_RE = re.compile(r"(INSERT\s+\w+[\s\S]*?\.)", re.IGNORECASE)
MODIFY_RE = re.compile(r"(MODIFY\s+\w+[\s\S]*?\.)", re.IGNORECASE)

# ===== Models =====
class SelectItem(BaseModel):
    table: str
    target_type: Optional[str] = None
    target_name: Optional[str] = None
    used_fields: List[str] = []
    suggested_fields: Optional[List[str]] = None
    suggested_statement: str

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = None
    code: Optional[str] = ""
    selects: List[SelectItem] = []

# ===== Helpers =====
def get_replacement_table(table: str) -> str:
    t_up = table.upper()
    if t_up in NO_VIEW_TABLES:
        return "ACDOCA"
    elif t_up in TABLE_MAPPING:
        return TABLE_MAPPING[t_up]["source"].split("/")[0]
    return table

def remediation_comment(table: str, stmt_type: str) -> str:
    t_up = table.upper()
    if stmt_type in ("UPDATE", "INSERT", "MODIFY", "DELETE"):
        return "/* NOTE: Compatibility view cannot be used for write ops. Use ACDOCA or source directly. */"
    if t_up in NO_VIEW_TABLES:
        return f"/* NOTE: {t_up} is obsolete in S/4HANA. Use ACDOCA directly. */"
    elif t_up in TABLE_MAPPING:
        src = TABLE_MAPPING[t_up]["source"]
        return f"/* NOTE: {t_up} is obsolete in S/4HANA. Use ACDOCA or view (source: {src}). */"
    return ""

def remediate_select(sel_text: str, table: str) -> str:
    rep_table = get_replacement_table(table)
    comment = remediation_comment(table, "SELECT")
    return re.sub(rf"\bFROM\s+{table}\b", f"FROM {rep_table} {comment}",
                  sel_text, flags=re.IGNORECASE).strip()

def remediate_other(stmt: str, stmt_type: str, table: str) -> str:
    rep_table = get_replacement_table(table)
    comment = remediation_comment(table, stmt_type)
    return re.sub(rf"({stmt_type}\s+(?:FROM\s+)?){table}\b",
                  rf"\1{rep_table} {comment}", stmt, flags=re.IGNORECASE).strip()

# ===== Parse selects =====
def parse_and_fill_selects(unit: Unit) -> List[SelectItem]:
    code = unit.code or ""
    findings: List[SelectItem] = []
    # SELECT
    for m in SELECT_RE.finditer(code):
        table = m.group("table")
        if table.upper() in TABLE_MAPPING or table.upper() in NO_VIEW_TABLES:
            findings.append(SelectItem(
                table=table,
                target_type="itab" if m.group("into_tab") else "wa",
                target_name=(m.group("into_tab") or m.group("into_wa")),
                used_fields=[],
                suggested_fields=None,
                suggested_statement=remediate_select(m.group("full"), table)
            ))
    # Others
    for stmt_type, pattern in [
        ("UPDATE", UPDATE_RE),
        ("DELETE", DELETE_RE),
        ("INSERT", INSERT_RE),
        ("MODIFY", MODIFY_RE)
    ]:
        for m in pattern.finditer(code):
            stmt_text = m.group(1).strip()
            table_match = re.search(rf"{stmt_type}\s+(?:FROM\s+)?(\w+)", stmt_text, re.IGNORECASE)
            if table_match:
                table = table_match.group(1)
                if table.upper() in TABLE_MAPPING or table.upper() in NO_VIEW_TABLES:
                    findings.append(SelectItem(
                        table=table,
                        target_type=None,
                        target_name=None,
                        used_fields=[],
                        suggested_fields=None,
                        suggested_statement=remediate_other(stmt_text, stmt_type, table)
                    ))
    return findings

# ===== Summariser =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    tables_count: Dict[str, int] = {}
    flagged = []
    for s in unit.selects:
        tbl_upper = s.table.upper()
        tables_count[tbl_upper] = tables_count.get(tbl_upper, 0) + 1
        flagged.append({"table": s.table, "reason": remediation_comment(s.table, "SELECT")})
    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_statements": len(unit.selects),
            "tables_frequency": tables_count,
            "note_2431747_flags": flagged
        }
    }

# ===== Prompt setup =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2431747. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code usage in light of SAP Note 2431747 (Obsolete FI/CO Tables in S/4HANA).

From S/4HANA onwards, the following tables are obsolete and replaced by ACDOCA or BSEG, sometimes as compatibility views:
{table_list}

We provide program/include/unit metadata, and statement analysis.

Your tasks:
1) Produce a concise **assessment** highlighting:
   - Which statements reference obsolete tables.
   - Why migration is needed.
   - Potential functional and data impact.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code in this unit for usage of obsolete FI/CO tables.
   - Replace with ACDOCA or BSEG as per mapping.
   - If statement is SELECT → allow view usage if supported; for writes → use direct table.
   - Add `TODO` comments for field mapping/manual remediation.
   - Output strictly in JSON with: original_code, remediated_code, changes[].

Return ONLY strict JSON:
{{
  "assessment": "<concise note 2431747 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "table_list": ", ".join(sorted(set(list(TABLE_MAPPING.keys()) + list(NO_VIEW_TABLES)))),
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API POST =====
@app.post("/assess-fico-migration")
async def assess_fico_migration(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        # Fill selects with regex parser
        u.selects = parse_and_fill_selects(u)
        # Get LLM output
        llm_out = llm_assess_and_prompt(u)
        obj = u.model_dump()
        obj.pop("selects", None)  # remove raw selects from output
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}