# Analyze compliance.txt for project TranscrevAI coding;
# Analyze fixes.txt and latest.txt for the last fixes made on app and latest implementations made on app;
# Analyze ".claude/" folder files to decide what will be the best approach when chatting with user;

# Using Gemini CLI for Large Codebase Analysis
  When analyzing large codebases or multiple files that might exceed context limits or spend a lot of tokens on analysis, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity, maximizing Claude Code's use and making it more effective and efficient.

  ## File and Directory Inclusion Syntax
  Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

  ### Examples:
  ```bash
  Single file analysis:
  gemini -p "@src/main.py Explain this file's purpose and structure"

  Multiple files:
  gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

  Entire directory:
  gemini -p "@src/ Summarize the architecture of this codebase"

  Multiple directories:
  gemini -p "@src/ @tests/ Analyze test coverage for the source code"

  Current directory and subdirectories:
  gemini -p "@./ Give me an overview of this entire project"

 Or use --all_files flag:
  gemini --all_files -p "Analyze the project structure and dependencies"

  Implementation Verification Examples

  Check if a feature is implemented:
  gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

  Verify authentication implementation:
  gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

  Check for specific patterns:
  gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

  Verify error handling:
  gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

  Check for rate limiting:
  gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

  Verify caching strategy:
  gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

  Check for specific security measures:
  gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

  Verify test coverage for features:
  gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

  When to Use Gemini CLI:
  Use gemini -p when:
  - Analyzing entire codebases or large directories;
  - Comparing multiple large files;
  - Need to understand project-wide patterns or architecture;
  - Current context window is insufficient for the task;
  - Working with files totaling more than 100KB;
  - Verifying if specific features, patterns, or security measures are implemented;
  - Checking for the presence of certain coding patterns across the entire codebase;

  Important Notes:
  - Paths in @ syntax are relative to your current working directory when invoking gemini;
  - The CLI will include file contents directly in the context;
  - No need for --yolo flag for read-only analysis;
  - Gemini's context window can handle entire codebases that would overflow Claude's context;
  - When checking implementations, be specific about what you're looking for to get accurate results;

# Triple-Validation Strategy for Complex Tasks
  How to use Gemini: on complex jobs that would be extremely token costly, use Gemini to maximize Claude Code's usability, maintaining efficiency by asking three different times for a resume of the topic/job and compare them. Go foward with the one that makes more sense to you, and feel free to ask me for help or my analysis.

  ## Triple Resume Strategy:
  For high-complexity implementations that could consume significant tokens:

  ### Step 1: Ask for 3 Different Summaries
  ```bash
  # Summary 1: Focus on specific file
  gemini -p "@src/target_file.py Analyze [problem] and give brief summary of fixes needed"

  # Summary 2: Focus on multiple related files
  gemini -p "@src/file1.py @src/file2.py What causes [problem] and how to fix it? Brief plan"

  # Summary 3: Focus on architecture/approach
  gemini -p "@src/ @lib/ Overall approach to solve [problem]. Implementation strategy"
  ```

  ### Step 2: Compare Consistency
  - If all 3 summaries align: proceed with implementation
  - If 2/3 align: use the majority approach
  - If all 3 differ: something is wrong, ask user for guidance

  ### Step 3: Implementation Decision
  - Choose the most robust/sustainable solution
  - Consider compliance with propositions.txt
  - Prefer architectural fixes over quick patches
  - Ask user if uncertain between approaches

  ### Benefits:
  - Reduces token waste on wrong implementations
  - Validates understanding before coding
  - Catches architectural issues early
  - Ensures sustainable solutions

  ### When to Use:
  - Complex refactoring tasks
  - Performance optimization problems
  - Integration between multiple systems
  - Architecture-changing implementations
  - When unsure about root cause vs symptoms