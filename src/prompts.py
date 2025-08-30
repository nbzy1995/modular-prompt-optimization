######################################
# Expert Persona
######################################

EXPERT_PERSONA_PROMPT = """ROLE: You are a domain expert with deep knowledge in the relevant field. Draw upon your expertise to provide the most accurate and comprehensive response possible."""

######################################
# Uncertainty Quantification
######################################

UNCERTAINTY_PROMPT = """CONFIDENCE ASSESSMENT: After providing your answer, explicitly state your confidence level as a percentage (0-100%). If your confidence is below 90%, clearly indicate which parts you are uncertain about."""


######################################
# Zeroshot Chain-of-Thought
######################################

COT_PROMPT = """REASONING PROCESS: Explicitly show your step-by-step reasoning process. Work through the problem methodically, showing each logical step."""

######################################
# Zeroshot Chain-of-Verification
######################################

COVE_PROMPT = """VERIFICATION PROCEDURE: Use the following verification approach:
1. INITIAL ANSWER: Provide your first response to the question
2. VERIFICATION QUESTIONS: Generate 3-5 specific verification questions to check your initial answer
3. VERIFICATION RESPONSES: Answer each verification question thoroughly
4. FINAL VERIFIED ANSWER: Based on the verification, provide your refined final answer

IMPORTANT: Even if the task asks for a specific format (like numbered lists), you must still show the complete verification process first, then provide the final answer in the requested format."""


######################################
# WIKIDATA
######################################

BASELINE_PROMPT_WIKI = """Answer the below question which is asking for a list of persons. Output should be a numbered list of maximum 10 persons and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Example Question: Who are some movie actors who were born in Boston?
Example Answer: 1. Donnie Wahlberg
2. Chris Evans
3. Mark Wahlberg
4. Ben Affleck
5. Uma Thurman
Example Question: Who are some football players who were born in Madrid?
Example Answer: 1. Sergio Ramos
2. Marcos Alonso
3. David De Gea
4. Fernando Torres

Example Question: Who are some politicians who were born in Washington?
Example Answer: 1. Barack Obama
2. Bill Clinton
3. Bil Sheffield
4. George Washington

Question: {question}
"""


######################################
# WIKIDATA CATEGORY
######################################

BASELINE_PROMPT_WIKI_CATEGORY = """Answer the below question which is asking for a list of entities (names, places, locations etc). Output should be a numbered list and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Example Question: Name some movies directed by Steven Spielberg.
Example Answer: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET

Example Question: Name some football stadiums from the Premier League.
Example Answer: 1. Old Trafford
2. Anfield
3. Stamford Bridge
4. Santiago Bernabeu

Question: {question}
"""


######################################
# MULTISPAN QA
######################################

BASELINE_PROMPT_MULTI_QA = """Answer the below question correctly and in a concise manner without much details. Only answer what the question is asked. NO ADDITIONAL DETAILS.

Question: {question}
"""