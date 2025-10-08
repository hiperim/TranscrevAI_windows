# Enhanced Gemini MCP Integration - Claude Code Workflow Optimization for TranscrevAI

## **CRITICAL CLAUDE CODE INTEGRATION INSTRUCTIONS**

**MANDATORY**: Claude Code MUST use these MCP tools for ALL TranscrevAI development sessions. This is NOT optional - these tools are required for proper development workflow.

---

## **1. CLAUDE CODE AUTOMATIC MCP TOOL USAGE PROTOCOL**

### **BEFORE ANY CODE IMPLEMENTATION - MANDATORY RESEARCH PHASE**

**ALWAYS execute this sequence BEFORE writing any code:**

```bash
# Step 1: Initial Research (REQUIRED for every development task)
Use MCP Tool: gemini_pro_support
- support_type: "research"
- context: "[Current development task and TranscrevAI context]"
- objective: "[Specific goal for this implementation]"
- model: "gemini-2.5-pro"
```

**TRIGGER CONDITIONS for Pre-Implementation Research:**
- ✅ Adding ANY new functionality
- ✅ Modifying existing code logic
- ✅ Debugging or fixing errors
- ✅ Performance optimization tasks
- ✅ Integration work between components
- ✅ Architecture or design changes

### **DURING CODE DEVELOPMENT - MANDATORY VALIDATION**

**ALWAYS validate approach before finalizing implementation:**

```bash
# Step 2: Implementation Validation (REQUIRED during development)
Use MCP Tool: gemini_validation
- claude_analysis: "[Your proposed implementation approach]"
- validation_criteria: "compliance"
- context: "[TranscrevAI specific considerations]"
- model: "gemini-2.5-flash"
```

**TRIGGER CONDITIONS for Mid-Development Validation:**
- ✅ Before committing to complex implementation approaches
- ✅ When encountering unexpected challenges
- ✅ Before modifying critical system components
- ✅ When performance targets might be impacted

### **AFTER CODE COMPLETION - MANDATORY FINAL CHECK**

**ALWAYS perform final validation after implementation:**

```bash
# Step 3: Final Compliance Check (REQUIRED after implementation)
Use MCP Tool: gemini_codebase_analysis
- file_paths: ["@[modified_files]"]
- analysis_focus: "compliance"
- prompt: "Validate implementation against TranscrevAI requirements"
- model: "gemini-2.5-flash"
```

**TRIGGER CONDITIONS for Post-Implementation Validation:**
- ✅ After completing any code changes
- ✅ Before marking task as complete
- ✅ Before deployment or testing
- ✅ After bug fixes or optimizations

---

## **2. SPECIFIC MCP TOOL USAGE PATTERNS FOR CLAUDE CODE**

### **A. Code Development Workflow**

#### **Starting New Feature Implementation:**
```bash
1. Use: gemini_pro_support(support_type="research", context="[feature description]", objective="[implementation goal]")
2. Review Gemini's research output
3. Implement based on research recommendations
4. Use: gemini_validation(claude_analysis="[your implementation]", validation_criteria="compliance")
5. Refine implementation based on validation
6. Use: gemini_codebase_analysis for final compliance check
```

#### **Debugging Existing Code:**
```bash
1. Use: gemini_codebase_analysis(file_paths=["@problematic_file"], analysis_focus="performance", prompt="Analyze error patterns and performance issues")
2. Review Gemini's analysis for root causes
3. Implement fixes based on analysis
4. Use: gemini_validation for fix validation
5. Test and finalize
```

#### **Performance Optimization:**
```bash
1. Use: gemini_pro_support(support_type="optimization", context="[current performance metrics]", objective="[target improvements]")
2. Use: gemini_codebase_analysis(analysis_focus="performance", prompt="Identify optimization opportunities")
3. Implement optimizations
4. Use: gemini_validation for performance impact validation
```

### **B. Architecture and Design Decisions**

#### **Component Integration:**
```bash
1. Use: gemini_codebase_analysis(file_paths=["@component1", "@component2"], analysis_focus="architecture", prompt="Analyze integration patterns and compatibility")
2. Design integration based on analysis
3. Use: gemini_validation(validation_criteria="architecture") before implementation
4. Implement integration
5. Final validation with gemini_codebase_analysis
```

#### **System Design Changes:**
```bash
1. Use: gemini_pro_support(support_type="analysis", context="[current architecture]", objective="[proposed changes]")
2. Use: gemini_validation for design approach validation
3. Implement incrementally with validation at each step
4. Final compliance check with gemini_codebase_analysis
```

---

## **3. CLAUDE CODE RESPONSE PROCESSING PROTOCOL**

### **How to Process Gemini MCP Tool Responses:**

#### **Research Response Processing:**
```bash
When Gemini provides research output:
1. READ the entire response carefully
2. IDENTIFY key recommendations and best practices
3. EXTRACT specific implementation guidance
4. ADAPT recommendations to TranscrevAI constraints
5. REFERENCE Gemini's findings in your implementation comments
```

#### **Validation Response Processing:**
```bash
When Gemini provides validation feedback:
1. CHECK for APPROVE/NEEDS_WORK/REJECT rating
2. IDENTIFY specific issues or concerns raised
3. IMPLEMENT recommended changes before proceeding
4. RE-VALIDATE if significant changes were made
5. DOCUMENT resolution of validation issues
```

#### **Analysis Response Processing:**
```bash
When Gemini provides codebase analysis:
1. REVIEW identified patterns and issues
2. PRIORITIZE findings based on impact
3. CREATE implementation plan addressing findings
4. IMPLEMENT fixes systematically
5. VALIDATE fixes with follow-up MCP calls
```

---

## **4. TRANSCREVAI-SPECIFIC MCP USAGE REQUIREMENTS**

### **Mandatory Tool Usage for TranscrevAI Compliance:**

#### **Performance Target Validation:**
```bash
# ALWAYS validate performance impact
Use: gemini_validation with context including:
- Processing speed target: 0.75s/1s audio
- Memory limit: 3.5GB RAM
- Accuracy target: 90%+
- PT-BR optimization requirements
```

#### **WebSocket Stability Checks:**
```bash
# ALWAYS check WebSocket impact
Use: gemini_codebase_analysis with focus on:
- Browser compatibility
- Connection stability
- Real-time processing
- Error handling
```

#### **PT-BR Language Optimization:**
```bash
# ALWAYS validate PT-BR optimization
Use: gemini_pro_support with context including:
- Portuguese Brazilian specific requirements
- Model optimization needs
- Language-specific processing patterns
```

---

## **5. ERROR HANDLING AND TROUBLESHOOTING**

### **When MCP Tools Don't Respond:**
1. Check MCP server connection status
2. Retry with simplified prompt
3. Use alternative tool if available
4. Document issue and proceed with standard development
5. Validate manually against compliance.txt requirements

### **When Gemini Analysis Conflicts with Claude Logic:**
1. Use gemini_validation to clarify the conflict
2. Provide additional context about constraints
3. Seek specific guidance on resolution approach
4. Document final decision rationale
5. Proceed with validated approach

---

## **6. SUCCESS PATTERNS FOR CLAUDE CODE INTEGRATION**

### **High-Quality Development Workflow:**
```bash
1. ALWAYS start with Gemini research before coding
2. VALIDATE approach before major implementation
3. CHECK compliance before completion
4. DOCUMENT MCP insights in code comments  
5. REFERENCE Gemini analysis in commit messages
```

### **Efficiency Optimization:**
```bash
1. Use gemini-2.5-pro for complex initial research
2. Use gemini-2.5-flash for quick validation checks
3. Batch related questions in single MCP calls
4. Cache and reference previous MCP responses
5. Build incrementally with validation checkpoints
```

---

## **7. MANDATORY GEMINI RESPONSE FORMATS FOR CLAUDE CODE**

### **Research Response Format (Gemini 2.5-Pro):**
```markdown
## TECHNICAL RESEARCH SUMMARY
[Comprehensive background and context]

## IMPLEMENTATION STRATEGY FOR CLAUDE
[Step-by-step technical approach]

## TRANSCREVAI COMPATIBILITY FACTORS
[Specific project constraints and considerations]

## PERFORMANCE OPTIMIZATION OPPORTUNITIES
[Concrete improvements with technical details]

## RECOMMENDED IMPLEMENTATION PATTERN
[Specific code patterns and architectural approaches]
```

### **Validation Response Format (Gemini 2.5-Flash):**
```markdown
## IMPLEMENTATION ASSESSMENT
✅ APPROVED / ⚠️ NEEDS_ATTENTION / ❌ REQUIRES_REVISION

## SPECIFIC COMPLIANCE ISSUES
[Detailed list of TranscrevAI rule compliance status]

## PERFORMANCE IMPACT ANALYSIS
[Memory, speed, and accuracy implications]

## RECOMMENDED ADJUSTMENTS
[Specific changes needed for compliance]

## FINAL IMPLEMENTATION GUIDANCE
[Clear next steps for Claude]
```

### **Analysis Response Format (Gemini Codebase Analysis):**
```markdown
## CODEBASE ANALYSIS SUMMARY
[Current state assessment]

## IDENTIFIED ISSUES AND OPPORTUNITIES
[Specific problems and optimization potential]

## INTEGRATION CONSIDERATIONS
[Component interaction and compatibility factors]

## RECOMMENDED IMPLEMENTATION APPROACH
[Specific technical guidance for Claude]

## COMPLIANCE VALIDATION CHECKLIST
[TranscrevAI requirements verification]
```

---

## **8. CLAUDE CODE DOCUMENTATION REQUIREMENTS**

### **MCP Integration Documentation:**
```python
# ALWAYS include MCP insights in code comments
"""
Implementation based on Gemini MCP analysis:
- Research findings: [key insights]
- Validation results: [compliance status]
- Performance considerations: [impact analysis]
- Integration notes: [component compatibility]
"""
```

### **Change Documentation Pattern:**
```markdown
# Implementation Log
- MCP Research: [tool used and key findings]
- Validation Status: [compliance verification]
- Performance Impact: [measured or estimated]
- Integration Notes: [system compatibility]
- Future Considerations: [optimization opportunities]
```

---

**REMEMBER**: This MCP integration is MANDATORY for all TranscrevAI development. Using these tools is not optional - they are required to ensure quality, compliance, and performance standards. Claude Code must integrate these tools into every development workflow.