"""LLM client for OpenAI GPT-4 integration"""

import os
from typing import Optional

from openai import OpenAI


class LLMClient:
    """Client for interacting with OpenAI GPT-4"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenAI API key. If None, will try to get from
                     environment or global config.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set it using set_api_key() or set OPENAI_API_KEY "
                "environment variable."
            )
        self.client = OpenAI(api_key=self.api_key)

    def generate_feature_code(
        self,
        df_info: str,
        metadata_info: str,
        target_column: Optional[str] = None,
        categorical_cols: Optional[list] = None,
        model: str = "gpt-4o",
        problem_description: Optional[str] = None,
        return_report: bool = False,
    ) -> str | tuple[str, str]:
        """
        Generate feature engineering code using GPT-4.

        Args:
            df_info: Information about the DataFrame (columns, dtypes,
                     sample data)
            metadata_info: Information from metadata DataFrame
            target_column: Name of the target/label column if available
            categorical_cols: List of categorical column names
            model: OpenAI model to use (default: "gpt-4o", alternatives:
                  "gpt-4-turbo", "gpt-3.5-turbo")
            problem_description: Optional description of the problem/use case
                                to provide additional context
            return_report: If True, also generate and return a feature report

        Returns:
            If return_report=False: Generated Python code for feature engineering
            If return_report=True: Tuple of (code, report) where report contains
                                  domain understanding and feature explanations
        """
        prompt = self._build_prompt(
            df_info,
            metadata_info,
            target_column,
            categorical_cols,
            problem_description,
            return_report,
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert data scientist specializing "
                            "in feature engineering for machine learning. "
                            "You understand domain context from metadata "
                            "descriptions and generate contextually relevant "
                            "features that are specifically designed to help "
                            "predict the target variable. Generate clean, "
                            "efficient Python code for creating new features "
                            "from existing numerical and categorical columns "
                            "in pandas DataFrames."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=4000 if return_report else 2000,
            )

            full_response = response.choices[0].message.content.strip()

            if return_report:
                # Extract report and code separately
                # Report should come first, then code
                report = ""
                code = ""

                # Try to find report section
                if "FEATURE REPORT" in full_response or "DOMAIN UNDERSTANDING" in full_response:
                    # Split by report markers
                    if "FEATURE REPORT" in full_response:
                        parts = full_response.split("FEATURE REPORT", 1)
                        if len(parts) > 1:
                            report_section = parts[1]
                            # Extract report until code section
                            if "```" in report_section:
                                report = report_section.split("```")[0].strip()
                            else:
                                report = report_section.strip()
                elif "---" in full_response:
                    # Try splitting by separator
                    parts = full_response.split("---", 1)
                    if len(parts) > 1:
                        report = parts[0].strip()
                        full_response = parts[1].strip()

                # Extract code block
                if "```python" in full_response:
                    code = full_response.split("```python")[1].split("```")[0].strip()
                elif "```" in full_response:
                    parts = full_response.split("```")
                    if len(parts) >= 3:
                        code = parts[1].strip()
                        if code.startswith("python"):
                            code = code[6:].strip()
                else:
                    # No code block, try to extract code lines
                    lines = full_response.split("\n")
                    code_lines = []
                    in_code = False
                    for line in lines:
                        if any(
                            line.strip().startswith(prefix)
                            for prefix in ["df[", "import ", "from ", "pd.", "np."]
                        ):
                            in_code = True
                        if (
                            in_code
                            or line.strip().startswith("df[")
                            or ("=" in line and "df" in line)
                        ):
                            code_lines.append(line)
                    code = "\n".join(code_lines).strip()

                # If report is empty, try to extract from beginning
                if not report:
                    # Take everything before code as report
                    if "```" in full_response:
                        report = full_response.split("```")[0].strip()
                    else:
                        # Try to find where code starts
                        code_start_markers = ["df[", "import ", "from "]
                        for marker in code_start_markers:
                            if marker in full_response:
                                idx = full_response.find(marker)
                                report = full_response[:idx].strip()
                                break

                # Clean code
                code = self._clean_code(code)

                # Validate
                if not code or len(code) < 10:
                    raw_content = full_response[:200]
                    raise RuntimeError(
                        "Generated code appears to be empty or invalid. "
                        f"Raw response: {raw_content}"
                    )

                return code, report
            else:
                # Original behavior - just extract code
                code = full_response

                # Extract code block if wrapped in markdown
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    # Handle generic code blocks
                    parts = code.split("```")
                    if len(parts) >= 3:
                        code = parts[1].strip()
                        # Remove language identifier if present
                        # (e.g., "python" at the start)
                        if code.startswith("python"):
                            code = code[6:].strip()

                # Clean up any remaining markdown or explanations
                code = self._clean_code(code)

                # Validate that we have actual code
                if not code or len(code) < 10:
                    raw_content = full_response[:200]
                    raise RuntimeError(
                        "Generated code appears to be empty or invalid. "
                        f"Raw response: {raw_content}"
                    )

                return code

        except Exception as e:
            raise RuntimeError(f"Error generating feature code: {str(e)}")

    def _clean_code(self, code: str) -> str:
        """Clean extracted code by removing non-code lines."""
        lines = code.split("\n")
        cleaned_lines = []
        in_code = False
        for line in lines:
            # Skip markdown headers, explanations, etc.
            if line.strip().startswith("#") and not in_code:
                # Allow comments in code, but skip if it's clearly
                # an explanation
                skip_words = ["here", "example", "note:", "important:"]
                if any(word in line.lower() for word in skip_words):
                    continue
            # Start collecting code when we see actual Python statements
            if any(
                line.strip().startswith(prefix)
                for prefix in ["df[", "import ", "from ", "pd.", "np."]
            ):
                in_code = True
            if in_code or line.strip().startswith("df[") or "=" in line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _build_prompt(
        self,
        df_info: str,
        metadata_info: str,
        target_column: Optional[str],
        categorical_cols: Optional[list] = None,
        problem_description: Optional[str] = None,
        return_report: bool = False,
    ) -> str:
        """Build the prompt for feature generation"""

        prompt = (
            f"""Generate Python code for feature engineering on a pandas """
            f"""DataFrame.

DATAFRAME INFORMATION:
{df_info}

METADATA INFORMATION (CRITICAL - USE THESE DESCRIPTIONS TO UNDERSTAND """
            f"""DOMAIN CONTEXT):
{metadata_info}
"""
        )

        if problem_description:
            prompt += (
                f"""
PROBLEM DESCRIPTION (ADDITIONAL CONTEXT):
{problem_description}

Use this problem description to better understand the business context and """
                f"""generate features that are specifically relevant to this use case.
"""
            )

        if target_column:
            prompt += (
                f"""
TARGET/LABEL COLUMN: {target_column}
IMPORTANT: The label_definition in the metadata above explains what we """
                f"""are trying to predict.
Use this understanding to generate features that are specifically """
                f"""relevant to predicting this target.
"""
            )

        if categorical_cols:
            cat_cols_str = ", ".join(categorical_cols)
            prompt += f"\nCATEGORICAL COLUMNS: {cat_cols_str}\n"

        prompt += """
CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. UNDERSTAND THE DOMAIN FROM METADATA:
   - Carefully read the column descriptions in the METADATA INFORMATION
     section above
   - Understand what each column represents in the domain context
   - Use this domain knowledge to generate features that make sense for
     this specific problem

2. UNDERSTAND THE PREDICTION TASK:
   - If a target column is specified, carefully read its label_definition
     in the metadata
   - Understand what you are trying to predict (e.g., customer churn,
     house price, disease diagnosis)
   - Generate features that are specifically relevant to predicting this
     target
   - Consider domain-specific relationships that would help predict the
     target

3. GENERATE CONTEXTUALLY RELEVANT FEATURES:
   - Base your feature engineering on the column descriptions and target
     definition
   - Create features that leverage domain knowledge (e.g., if predicting
     income, create income-to-expense ratios)
   - Generate interactions between columns that make sense in the domain
     context
   - Consider what a domain expert would create as features for this
     specific prediction task

4. NUMERICAL FEATURE TECHNIQUES (apply based on domain relevance):
   - Mathematical transformations (log, sqrt, square, etc.) - use when
     they make domain sense
   - Statistical aggregations (mean, std, min, max across columns) - for
     related numerical columns
   - Interactions between numerical columns - create ratios, products,
     differences that are meaningful
   - Polynomial features - when non-linear relationships are expected
   - Binning/discretization - for creating categorical-like features from
     continuous variables
   - Time-based features if datetime columns exist

5. CATEGORICAL FEATURE TECHNIQUES (apply based on cardinality and domain
   relevance):
   - One-hot encoding (pd.get_dummies) for low cardinality categories
     (<=10 unique values)
   - Target encoding (mean encoding) if target column is available -
     especially useful for high cardinality
   - Frequency encoding (count of each category) - for understanding
     category prevalence
   - Label encoding for ordinal categories where order matters
   - Interaction features between categorical and numerical columns -
     create group statistics
   - Group statistics (mean, median, std, etc. of numerical columns
     grouped by category)
   - Rare category grouping (group categories with low frequency into
     'Other' category)

6. CODE REQUIREMENTS (CRITICAL):
   - Return ONLY the Python code, no explanations, no markdown, no
     comments outside code
   - The code MUST use 'df' as the DataFrame variable name (e.g.,
     df['new_feature'] = ...)
   - The code MUST create new columns using df['new_feature_name'] = ...
   - DO NOT use print() statements or any output - only create columns
   - DO NOT reassign df (e.g., df = ...) - only modify it in place
   - Use pandas and numpy operations
   - Handle potential errors gracefully (e.g., division by zero, log of
     negative numbers, missing values)
   - For categorical encoding, handle unseen categories appropriately
     (use fillna or default values)
   - Use descriptive column names that reflect what the feature
     represents
   - Generate at least 3-5 meaningful features based on the domain
     context
   - Each line should create a new column: df['feature_name'] = ...

7. PRIORITIZE FEATURES RELEVANT TO THE TARGET:
   - If a target is specified, prioritize features that directly relate
     to predicting it
   - Create features that capture relationships between predictors and
     the target
   - Consider what features would be most informative for the specific
     prediction task

Generate the feature engineering code:"""

        if return_report:
            prompt += """

IMPORTANT: After generating the code, provide a FEATURE REPORT with the following structure:

FEATURE REPORT
==============

1. DOMAIN UNDERSTANDING:
   - Summarize your understanding of the problem domain based on the metadata and problem description
   - Explain the business context and what we're trying to predict
   - Describe key relationships and patterns you identified in the data

2. GENERATED FEATURES EXPLANATION:
   For each feature you generated, provide:
   - Feature Name: [name of the feature]
   - Description: [what this feature represents]
   - Rationale: [why this feature is useful for predicting the target]
   - Domain Relevance: [how this feature relates to the business problem]

Format the report clearly with sections and bullet points for readability.
"""

        return prompt

        return prompt
