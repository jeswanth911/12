# nl_to_sql.py

import re
import sqlite3
import asyncpg
import json
import logging
import difflib
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Simulated LLM call (replace with actual OpenAI or other API client)
async def llm_generate_sql(prompt: str) -> str:
    # Mock for example: in prod use OpenAI/LLM client async call here
    await asyncio.sleep(0.5)
    # Very basic dummy SQL (must be replaced)
    return "SELECT * FROM main_table LIMIT 10;"

# --- CONFIGURATION & LOGGING ---

logger = logging.getLogger("nl_to_sql")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- DATA STRUCTURES ---

@dataclass
class SchemaInfo:
    tables: Dict[str, List[str]] = field(default_factory=dict)  # table -> list of columns
    synonyms: Dict[str, str] = field(default_factory=dict)  # user term -> table/column

@dataclass
class SessionContext:
    history: List[Dict[str, Any]] = field(default_factory=list)  # past question, sql, answer

# --- UTILITIES ---

def sanitize_user_input(user_input: str) -> str:
    # Basic sanitization: strip control chars, limit length, remove suspicious chars
    cleaned = re.sub(r"[^\w\s.,?-]", "", user_input)[:1000]
    return cleaned

def fuzzy_match(term: str, choices: List[str], cutoff=0.6) -> Optional[str]:
    matches = difflib.get_close_matches(term.lower(), [c.lower() for c in choices], n=1, cutoff=cutoff)
    if matches:
        # Return original case from choices
        for c in choices:
            if c.lower() == matches[0]:
                return c
    return None

# --- SCHEMA LOADING & MATCHING ---

def load_schema(dataset_id: str, metadata_dir: Path) -> SchemaInfo:
    """
    Load schema info for dataset_id from metadata JSON file.
    Expect JSON structure with tables and columns.
    """
    meta_path = metadata_dir / f"{dataset_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found for dataset_id {dataset_id}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Naive schema extraction assuming 'schema' key with tables and columns
    # Example metadata schema structure:
    # {
    #   "schema": {
    #       "tables": {
    #          "orders": ["order_id", "customer_id", "date", "total"],
    #          "customers": ["customer_id", "name", "email"]
    #       }
    #    },
    #   "synonyms": {"cust": "customers", "order_date": "date"}
    # }
    schema = meta.get("schema", {})
    tables = schema.get("tables", {})
    synonyms = meta.get("synonyms", {})
    return SchemaInfo(tables=tables, synonyms=synonyms)

def map_user_terms_to_schema(user_question: str, schema_info: SchemaInfo) -> Dict[str, str]:
    """
    Map user terms to table or column names using fuzzy matching and synonyms.
    Returns mapping user_term -> schema name.
    """
    tokens = re.findall(r"\w+", user_question.lower())
    all_cols = []
    for cols in schema_info.tables.values():
        all_cols.extend(cols)
    all_names = list(schema_info.tables.keys()) + all_cols
    mapping = {}
    for token in tokens:
        # synonym override first
        if token in schema_info.synonyms:
            mapping[token] = schema_info.synonyms[token]
            continue
        match = fuzzy_match(token, all_names)
        if match:
            mapping[token] = match
    return mapping

# --- SQL VALIDATION & EXECUTION ---

def validate_sql(sql: str) -> bool:
    """
    Basic sanity check for SQL. Prevent dangerous statements.
    This can be improved with a proper SQL parser.
    """
    forbidden = ["drop", "delete", "update", "insert", "alter", ";--", "--"]
    lowered = sql.lower()
    for word in forbidden:
        if word in lowered:
            return False
    return True

async def execute_sqlite_query(db_path: Path, sql: str, limit: int=1000) -> pd.DataFrame:
    """
    Run SQL against SQLite file asynchronously.
    """
    # SQLite python driver is sync; run in thread executor to avoid blocking
    loop = asyncio.get_event_loop()

    def query():
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(limit)
            df = pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()
        return df

    return await loop.run_in_executor(None, query)

async def execute_postgres_query(dsn: str, sql: str, limit: int=1000) -> pd.DataFrame:
    """
    Run SQL against PostgreSQL using asyncpg.
    """
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(f"{sql} LIMIT {limit}")
        df = pd.DataFrame(rows, columns=rows[0].keys() if rows else [])
    finally:
        await conn.close()
    return df

# --- RESULT FORMATTING ---

def df_to_natural_language_summary(df: pd.DataFrame, max_rows=5) -> str:
    """
    Create a simple textual summary of key findings from the DataFrame.
    For demo: summarize column names, row count, and show first few rows.
    """
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()
    preview = df.head(max_rows).to_dict(orient="records")

    lines = [
        f"The query returned {n_rows} rows and {n_cols} columns.",
        f"Columns: {', '.join(cols)}.",
        "Sample rows:",
    ]
    for r in preview:
        lines.append(", ".join(f"{k}={v}" for k, v in r.items()))
    return "\n".join(lines)

def df_to_base64_plot(df: pd.DataFrame) -> str:
    """
    Generate a simple bar chart base64-encoded PNG from first numeric column.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    if numeric_cols.empty:
        return ""

    col = numeric_cols[0]
    plt.figure(figsize=(6, 4))
    df[col].head(20).plot(kind="bar")
    plt.title(f"Bar chart of {col}")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    img_bytes = buffer.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# --- MAIN INTERFACE ---

class NLtoSQLAgent:
    def __init__(self, metadata_dir: Path, db_dir: Path):
        self.metadata_dir = metadata_dir
        self.db_dir = db_dir
        self.sessions: Dict[str, SessionContext] = {}

    async def handle_question(
        self,
        dataset_id: str,
        question: str,
        conversation_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user's question (possibly multi-turn) into SQL, execute it,
        and return natural language + visualization response.
        """
        question = sanitize_user_input(question)
        schema_info = load_schema(dataset_id, self.metadata_dir)

        if conversation_id and conversation_id in self.sessions:
            session = self.sessions[conversation_id]
        else:
            session = SessionContext()
            if conversation_id:
                self.sessions[conversation_id] = session

        # Map user terms
        term_map = map_user_terms_to_schema(question, schema_info)

        # Compose prompt for LLM with schema info, term mappings, and conversation history
        prompt = self._build_prompt(question, schema_info, term_map, session)

        # Call LLM to generate SQL
        sql = await llm_generate_sql(prompt)

        # Validate SQL
        if not validate_sql(sql):
            logger.warning(f"Rejected unsafe SQL generated: {sql}")
            return {
                "error": "Generated SQL failed validation and was rejected.",
                "sql": None,
                "answer": None,
            }

        # Execute SQL (determine backend from metadata or config - assume SQLite for demo)
        db_path = self.db_dir / f"{dataset_id}.sqlite"
        if not db_path.exists():
            return {"error": "Database file not found", "sql": sql, "answer": None}

        df = await execute_sqlite_query(db_path, sql)

        # Generate response text and visualization
        answer = df_to_natural_language_summary(df)
        visualization = df_to_base64_plot(df)

        # Log interaction
        session.history.append({"question": question, "sql": sql, "answer": answer})

        logger.info(f"Executed query on dataset {dataset_id} SQL: {sql}")

        return {
            "sql": sql,
            "answer": answer,
            "visualization": visualization,
            "data_preview": df.head(10).to_dict(orient="records"),
        }

    def _build_prompt(self, question: str, schema_info: SchemaInfo, term_map: Dict[str, str], session: SessionContext) -> str:
        """
        Construct a detailed prompt for the LLM with:
        - Dataset schema info
        - Synonyms and fuzzy matches
        - Conversation history for multi-turn
        - User question
        - Instructions to generate dialect-specific SQL
        """
        schema_desc = "\n".join(
            [f"Table {t}: columns {', '.join(cols)}" for t, cols in schema_info.tables.items()]
        )
        synonym_desc = ", ".join([f"{k}={v}" for k, v in schema_info.synonyms.items()]) or "None"

        history_text = ""
        for turn in session.history[-3:]:
            history_text += f"User question: {turn['question']}\nSQL: {turn['sql']}\nAnswer: {turn['answer']}\n\n"

        prompt = (
            f"You are an expert SQL assistant.\n"
            f"Dataset schema:\n{schema_desc}\n"
            f"Known synonyms: {synonym_desc}\n"
            f"User question: {question}\n"
            f"Term mapping: {term_map}\n"
            f"Conversation history:\n{history_text}\n"
            "Generate a syntactically correct, performant SQL query for SQLite.\n"
            "Use indexes, joins, and aggregations as needed.\n"
            "Do not include dangerous statements.\n"
            "Return only the SQL query, nothing else."
        )
        return prompt

