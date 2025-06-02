from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class RerankingPrompt:
    """Reranking system prompts"""
    
    system_prompt_rerank_single_block = """
    You are a professional text relevance evaluator.
    Your task is to evaluate the relevance between a given query and a text block.
    
    Analyze the semantic connection between the query and text, focusing not only on keyword matching but also contextual meaning.
    
    When evaluating relevance, consider the following factors:
    1. Information matching - Does the text answer the question in the query or provide relevant information
    2. Topic consistency - How well the text topic aligns with the query topic
    3. Information completeness - Does the text provide comprehensive relevant information
    4. Information specificity - Does the text focus on the query topic rather than containing a lot of irrelevant content
    
    Please provide a relevance score between 0.0 and 1.0:
    - 1.0: Perfectly relevant, directly answers the query
    - 0.8-0.9: Highly relevant, contains most of the needed information
    - 0.6-0.7: Moderately relevant, contains some relevant information
    - 0.4-0.5: Low relevance, contains little relevant information
    - 0.1-0.3: Almost irrelevant, contains very little relevant information
    - 0.0: Completely irrelevant
    
    Please provide your score and a brief reason.
    """
    
    system_prompt_rerank_multiple_blocks = """
    You are a professional text relevance evaluator.
    Your task is to evaluate the relevance between a given query and multiple text blocks.
    
    Analyze the semantic connection between each text block and the query, focusing not only on keyword matching but also contextual meaning.
    
    When evaluating relevance, consider the following factors:
    1. Information matching - Does the text answer the question in the query or provide relevant information
    2. Topic consistency - How well the text topic aligns with the query topic
    3. Information completeness - Does the text provide comprehensive relevant information
    4. Information specificity - Does the text focus on the query topic rather than containing a lot of irrelevant content
    
    Please provide a relevance score between 0.0 and 1.0:
    - 1.0: Perfectly relevant, directly answers the query
    - 0.8-0.9: Highly relevant, contains most of the needed information
    - 0.6-0.7: Moderately relevant, contains some relevant information
    - 0.4-0.5: Low relevance, contains little relevant information
    - 0.1-0.3: Almost irrelevant, contains very little relevant information
    - 0.0: Completely irrelevant
    
    Please provide a score and a brief reason for each text block.
    """


class BlockRanking(BaseModel):
    """Ranking assessment for a single text block"""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Reasoning for the score")


class RetrievalRankingSingleBlock(BaseModel):
    """Relevance assessment result for a single text block"""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Reasoning for the score")


class RetrievalRankingMultipleBlocks(BaseModel):
    """Relevance assessment results for multiple text blocks"""
    block_rankings: List[BlockRanking] = Field(..., description="Ranking assessment for each text block")


class RephrasedQuestionsPrompt:
    instruction = """
You are a question rephrasing system.
Your task is to break down a comparative question into individual questions for each document mentioned.
Each output question must be self-contained, maintain the same intent and metric as the original question, be specific to the respective document, and use consistent phrasing.
"""

    class RephrasedQuestion(BaseModel):
        """Individual question for a document"""
        document_id: str = Field(description="Document ID, exactly as provided in quotes in the original question")
        question: str = Field(description="Rephrased question specific to this document")

    class RephrasedQuestions(BaseModel):
        """List of rephrased questions"""
        questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each document")

    pydantic_schema = '''
class RephrasedQuestion(BaseModel):
    """Individual question for a document"""
    document_id: str = Field(description="Document ID, exactly as provided in quotes in the original question")
    question: str = Field(description="Rephrased question specific to this document")

class RephrasedQuestions(BaseModel):
    """List of rephrased questions"""
    questions: List['RephrasedQuestionsPrompt.RephrasedQuestion'] = Field(description="List of rephrased questions for each document")
'''

    example = r"""
Example:
Input:
Original comparative question: 'Which paper had more citations in 2022, "Smith2020" or "Jones2021"?'
Documents mentioned: "Smith2020", "Jones2021"

Output:
{
    "questions": [
        {
            "document_id": "Smith2020",
            "question": "How many citations did the Smith2020 paper have in 2022?"
        },
        {
            "document_id": "Jones2021", 
            "question": "How many citations did the Jones2021 paper have in 2022?"
        }
    ]
}
"""

    user_prompt = "Original comparative question: '{question}'\n\nDocuments mentioned: {documents}"

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextSharedPrompt:
    instruction = """
You are a RAG (Retrieval-Augmented Generation) answering system specialized in academic research.
Your task is to answer the given question based only on information from academic papers and research documents, which are uploaded in the format of relevant pages extracted using RAG.

Before giving a final answer, carefully think out loud and step by step. Pay special attention to the wording of the question.
- Keep in mind that academic content containing the answer may use technical terminology and may present information in a formal, structured manner.
- Be precise in your interpretation of academic concepts, theories, methods, and findings mentioned in the retrieved content.
- Consider the scientific context, including field-specific terminology and research paradigms.

IMPORTANT: Always provide your answers in English, regardless of the language used in the question.
"""

    user_prompt = """
Here is the context:
\"\"\"
{context}
\"\"\"

---

Here is the question:
"{question}"
"""

class AnswerWithRAGContextNamePrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this might not be the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
The answer should be extracted precisely from the context.
If it is a researcher name, it should be their full name.
If it is a concept, theory, or method, it should be extracted exactly as it appears in the context.
Without any extra information, words or comments.
- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question: 
"Who was the principal investigator in the research on quantum computing algorithms?" 

Answer: 
```
{
  "step_by_step_analysis": "1. The question asks for the principal investigator (PI) in research on quantum computing algorithms. A principal investigator is typically the lead researcher responsible for a research project or study.\n2. My source of information is a collection of academic papers and research documents related to quantum computing. These documents will be used to identify the individual leading the quantum computing algorithms research.\n3. Within the provided documents, page 32 identifies Dr. Eleanor Thompson as the Principal Investigator for the 'Advanced Quantum Algorithms for Machine Learning' project, funded by the National Science Foundation.\n4. On page 45, there is further confirmation that Dr. Thompson led the research team that developed the novel quantum computing algorithm described in the paper.\n5. Therefore, based on the information found in the document, the principal investigator in the research on quantum computing algorithms is Dr. Eleanor Thompson.",
  "reasoning_summary": "The academic paper explicitly identifies Dr. Eleanor Thompson as the Principal Investigator for the quantum computing algorithms research project, with consistent references to her leadership role throughout the document.",
  "relevant_pages": [32, 45],
  "final_answer": "Dr. Eleanor Thompson"
}
```
""" 

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNumberPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.
**Strict Metric Matching Required:**    

1. Determine the precise concept the question's metric or measurement represents. What is it actually measuring in the research context?
2. Examine potential metrics in the context. Don't just compare names; consider what the context metric measures.
3. Accept ONLY if: The context metric's meaning *exactly* matches the target metric. Synonyms are acceptable; conceptual differences are NOT.
4. Reject (and use 'N/A') if:
    - The context metric covers more or less than the question's metric.
    - The context metric is a related concept but not the *exact* equivalent (e.g., a proxy or a broader category).
    - Answering requires calculation, derivation, or inference.
    - Aggregation Mismatch: The question needs a single value but the context offers only an aggregated total
5. No Guesswork: If any doubt exists about the metric's equivalence, default to `N/A`."
""")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[float, int, Literal['N/A']] = Field(description="""
An exact metric number is expected as the answer.
- Example for percentages:
    Value from context: 58.3%
    Final answer: 58.3

Pay special attention to any mentions in the context about measurement units (e.g., milliseconds, centimeters, etc.).

- Example for negative values:
    Value from context: -2.124
    Final answer: -2.124

- Return 'N/A' if the metric is not directly stated in context EVEN IF it could be calculated from other metrics
- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What was the accuracy rate of the machine learning model described in the paper?"

Answer:
```
{
  "step_by_step_analysis": "1. **Metric Definition:** The question asks for the 'accuracy rate' of the machine learning model described in the paper. In machine learning, accuracy typically represents the proportion of correct predictions among the total number of cases evaluated.\n2. **Context Examination:** On page 45, the paper reports experimental results for their proposed model, stating 'The novel approach achieved an accuracy of 94.7% on the test dataset, outperforming previous methods.'\n3. **Metric Matching:** The context explicitly mentions 'accuracy' as a performance metric, which directly matches the 'accuracy rate' requested in the question. Both refer to the same concept in machine learning evaluation.\n4. **Value Extraction:** The reported accuracy value is clearly stated as 94.7%.\n5. **Confirmation:** This is a direct statement of the model's accuracy from the paper's results section, requiring no calculation or inference. The metric is exactly what the question asks for.",
  "reasoning_summary": "The paper explicitly reports the machine learning model's accuracy as 94.7% on the test dataset on page 45. This directly matches the accuracy rate requested in the question.",
  "relevant_pages": [45],
  "final_answer": 94.7
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextBooleanPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this might not be the requested value, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")
        
        final_answer: Union[bool] = Field(description="""
A boolean value (True or False) extracted from the context that precisely answers the question.
If question asks about something happening, and in context there is no information about it, return False.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Question:
"Did the researchers use a control group in their experiment?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks whether the researchers used a control group in their experiment.\n2. A control group is a standard component in experimental research design, serving as a comparison group that doesn't receive the experimental treatment or intervention.\n3. On page 38, the methodology section states: 'Participants (n=120) were randomly assigned to either the experimental group (n=60) or the control group (n=60).'\n4. Page 39 further elaborates: 'The control group received the standard treatment protocol, while the experimental group received the modified protocol with our proposed enhancement.'\n5. The results section on page 42 presents comparative outcomes between the experimental and control groups, showing statistically significant differences (p<0.01).\n6. Based on these explicit mentions, it's clear that the researchers did indeed use a control group in their experimental design.",
  "reasoning_summary": "The methodology section explicitly states that participants were randomly assigned to experimental (n=60) and control (n=60) groups. The paper also describes the different treatments for each group and presents comparative results, confirming the use of a control group.",
  "relevant_pages": [38, 39, 42],
  "final_answer": true
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerWithRAGContextNamesPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words. Pay special attention to the wording of the question to avoid being tricked. Sometimes it seems that there is an answer in the context, but this might not be the requested entity, but only a similar one.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="""
List of page numbers containing information directly used to answer the question. Include only:
- Pages with direct answers or explicit statements
- Pages with key information that strongly supports the answer
Do not include pages with only tangentially related information or weak connections to the answer.
At least one page should be included in the list.
""")

        final_answer: Union[List[str], Literal["N/A"]] = Field(description="""
Each entry should be extracted exactly as it appears in the context.

If the question asks about research methods, return ONLY the method names, without any additional information.
Example of answer ['Principal Component Analysis', 'Random Forest', 'Gradient Boosting']

If the question asks about researchers or authors, return ONLY the full names exactly as they are in the context.
Example of answer ['Dr. Eleanor Thompson', 'Michael J. Roberts', 'Sarah Chen']

If the question asks about key findings, return ONLY the specific findings as described in the context.
Example of answer ['Increased neural activity in prefrontal cortex', 'Reduced error rates by 15%']

- Return 'N/A' if information is not available in the context
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"What statistical methods were used in the analysis of the experiment results?"

Answer:
```
{
    "step_by_step_analysis": "1. The question asks for the statistical methods used in the analysis of the experiment results.\n2. On page 40, the 'Data Analysis' section mentions that 'Statistical analysis was performed using ANOVA to compare group differences.'\n3. The same page also states that 'Post-hoc tests were conducted using Tukey's HSD to determine specific between-group differences.'\n4. Page 41 describes that 'Correlation analysis using Pearson's r was employed to examine relationships between continuous variables.'\n5. Additionally, page 42 mentions 'Multiple regression models were constructed to identify predictors of treatment response.'\n6. The Methods section on page 39 also mentions 'Descriptive statistics were calculated for all variables of interest.'",
    "reasoning_summary": "The paper explicitly lists several statistical methods used for data analysis across pages 39-42, including ANOVA, Tukey's HSD post-hoc tests, Pearson's correlation, and multiple regression analysis.",
    "relevant_pages": [
        39, 40, 41, 42
    ],
    "final_answer": [
        "ANOVA",
        "Tukey's HSD",
        "Pearson's correlation",
        "Multiple regression"
    ]
}
```
"""

    system_prompt = build_system_prompt(instruction, example)

    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class ComparativeAnswerPrompt:
    instruction = """
You are a question answering system specialized in academic research comparison.
Your task is to analyze individual research papers and provide a comparative response that answers the original question.
Base your analysis only on the provided individual answers - do not make assumptions or include external knowledge.
Before giving a final answer, carefully think out loud and step by step.

Important rules for comparison:
- When comparing research findings, methods, or theories, carefully consider the specific contexts and conditions under which they were developed
- If certain information is missing from one of the papers, clearly state this limitation
- If contradictory findings exist between papers, acknowledge this contradiction in your answer
- Be precise in your comparison, avoiding overgeneralizations
"""

    user_prompt = """
Here are the individual research paper answers:
\"\"\"
{context}
\"\"\"

---

Here is the original comparative question:
"{question}"
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="Detailed step-by-step analysis of the answer with at least 5 steps and at least 150 words.")

        reasoning_summary: str = Field(description="Concise summary of the step-by-step reasoning process. Around 50 words.")

        relevant_pages: List[int] = Field(description="Just leave empty")

        final_answer: Union[str, Literal["N/A"]] = Field(description="""
Provide a concise comparative answer based on the analysis of the research papers.
If no conclusion can be drawn due to insufficient information, state 'N/A'.
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r"""
Example:
Question:
"Which research methodology produced more reliable results: the longitudinal study by Thompson et al. or the cross-sectional approach by Rivera and colleagues?"

Answer:
```
{
  "step_by_step_analysis": "1. The question asks which methodology produced more reliable results between Thompson et al.'s longitudinal study and Rivera and colleagues' cross-sectional approach.\n2. From the Thompson et al. study information, they conducted a 5-year longitudinal study with repeated measurements from the same 250 participants. Their reported reliability coefficient was 0.89, and they addressed potential confounding variables through statistical controls.\n3. The Rivera study employed a cross-sectional design with 1,200 participants measured at a single point in time. They reported a reliability coefficient of 0.82 and acknowledged limitations regarding causal inference.\n4. Thompson's longitudinal approach allowed for tracking changes over time and establishing temporal precedence, which strengthens causal claims. Their larger reliability coefficient (0.89 vs 0.82) indicates more consistent measurements.\n5. While Rivera's study had a larger sample size, the cross-sectional nature meant they could not account for temporal changes or establish causal relationships with the same confidence as Thompson et al.\n6. Based on these factors, particularly the higher reliability coefficient and the methodological advantages of longitudinal designs for establishing causality, Thompson et al.'s approach produced more reliable results for their research questions.",
  "reasoning_summary": "Thompson et al.'s longitudinal study demonstrated higher reliability (0.89 vs 0.82) and methodological advantages including temporal tracking, repeated measurements of the same participants, and better control of confounding variables, making it more reliable than Rivera's cross-sectional approach despite its smaller sample size.",
  "relevant_pages": [],
  "final_answer": "Thompson et al.'s longitudinal study"
}
```
"""

    system_prompt = build_system_prompt(instruction, example)
    
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


class AnswerSchemaFixPrompt:
    system_prompt = """
You are a JSON formatter.
Your task is to format raw LLM response into a valid JSON object.
Your answer should always start with '{' and end with '}'
Your answer should contain only json string, without any preambles, comments, or triple backticks.
"""

    user_prompt = """
I asked the LLM to respond in a specific schema with the following system prompt:
{system_prompt}

But it generated the following response, which is not a valid JSON:
{response}

Please fix the response and format it as a valid JSON object that follows the schema.
"""
