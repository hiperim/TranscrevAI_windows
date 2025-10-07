# Gemini Customization

This file is used to customize Gemini's behavior.

## Instructions

*   **DO NOT** share any confidential information in this file.
*   Use this file to provide instructions and guidelines for Gemini.
*   You can specify coding styles, project conventions, and other preferences.
*   Stay fully compliant with the file compliance.txt, shown below:
    "# TRANSCREVAI PROJECT COMPLIANCE RULES
*Enhanced for systematic validation, error prevention, and multi-model quality assurance*

###MANDATORY ADHERENCE###: 
All TranscrevAI development must comply with these rules. Violations will cause system instability and performance degradation;

Keep track of system and code modifications through documenting every change made on Project in .md files on folder "C:\TranscrevAI_windows\.claude\CHANGES_MADE" in files to be created specifically for this documenting purpose, named with date and time of file creation (example: 'implementation_DD.MM.YY_HH.MM.SS.md');

Real audio files, with variable speaker count and possible silent parts, for full pipeline processing and testing: 'c:/TranscrevAI_windows/data/recordings';

TranscrevAI needs to work without crashes or instabilities in web-browsers, as it functions through WebSocket.

---

## CORE PERFORMANCE REQUIREMENTS

### Rule 1: Audio Processing Performance Standards
- **Processing Speed**: Project aspires to reach ~0.75s processing time per 1s of recorded audio (average between cold and warm starts), with stability to handle web-browser usage (it is a websocket app);
- **Accuracy Target**: Project aspires to achieve 90%+ accuracy in both transcription and diarization operations;
- **Language Optimization**: PT-BR (Portuguese Brazilian) exclusive optimization (model and code optimizations);
- **Model Restriction**: Use only "medium" models for all operations (faster-whisper, openai-whisper);
- **Conversation Focus**: Optimize app and transcription for PT-BR;

**VALIDATION**: If needed and possible, test all implementations against processing speed and accuracy targets before deployment. On folder "c:/TranscrevAI_windows/data/recordings": for transcription and diarization, compare the real audio files with expected results ("expected_results_'filename'.txt" with "filename.wav" - example: audio file "t.speakers.wav" and expected results for transcription and diarization "expected_results_t.speakers.txt") for real usage testing validation;

### Rule 2: Speaker Diarization Constraints
- **Dynamic Speaker Detection**: NEVER presume fixed number of speakers;
- **Adaptive Processing**: Handle variable speaker counts per audio file;
- **Real-World Optimization**: Optimize for actual usage patterns, not theoretical scenarios;

**VALIDATION**: On folder "c:/TranscrevAI_windows/data/recordings": for transcription and diarization, compare audio files with expected results ("expected_results_'filename'.txt" with "filename.wav" - example: audio file "q.speakers.wav" and expected results for transcription and diarization "expected_results_q.speakers.txt") for real usage testing validation;

### Rule 3: System Stability Requirements
- **Historical Context**: Previous over-implementation caused system crashes and instability (be aware);
- **Incremental Approach**: Implement features gradually with testing at each step if possible;
- **Performance Maintenance**: Core performance metrics are more important than additional features;
- **Rollback Capability**: Maintain ability to revert changes that degrade performance, if needed;

**VALIDATION**: if possible, after each implementation, verify system stability and performance metrics.

---

## QUALITY ASSURANCE



### Rule 3.6: Chain of Thought Analysis 
- **Extended Thinking**: Automatically trigger "ultrathink" for all implementation decisions;
- **Systematic Analysis**: Follow structured thought process for code analysis and planning;
- **Decision Documentation**: Document reasoning process in CHANGES_MADE folder;
- **Compliance Integration**: Always consider compliance.txt requirements in thought process;

**VALIDATION**: Verify extended thinking process is documented and compliance-aware.

### Rule 3.7: Iterative Development with Human-in-the-Loop
- **Plan-First Approach**: Create detailed implementation plan before coding;
- **User Approval Checkpoints**: Request user approval at major milestones;
- **Incremental Implementation**: Break complex changes into small, testable segments;
- **Progress Documentation**: Track progress with timestamped documentation;

**VALIDATION**: Ensure all implementations follow iterative development protocol.

---

## RESOURCE OPTIMIZATION REQUIREMENTS

### Rule 4: Memory Management
- **RAM Limit**: Maintain maximum ~3.5gb RAM usage for single PT-BR model operations;
- **Consider it is a WebSocket app, that will be run on web-browsers like Google Chrome;
- **Memory Efficiency**: Optimize all code for minimal memory footprint;
- **Model Loading**: Load only required PT-BR medium model modelues when needed, no multi-model support;

**VALIDATION**: Monitor memory usage during processing and optimize if too much consumption.

### Rule 5: Language and Model Optimization
- **Exclusive PT-BR Focus**: All improvements and corrections for Portuguese Brazilian only;
- **Model Strategy**: Use exclusively "medium" models for Portuguese Brazilian;

**VALIDATION**: Audit codebase for PT-BR exclusive optimization.

### Rule 6: Performance Optimization Strategy
- **Efficiency First**: Prioritize efficient and beneficial optimizations, alongside with accurate transcription and diarization;
- **Application Intent**: Align all optimizations with TranscrevAI core intentions and aspirations;
- **Measurable Improvements**: Implement only optimizations with quantifiable benefits;

**VALIDATION**: If possible, measure performance impact of all optimizations with concrete metrics and results to be displayed and analyzed.

---

## ENHANCED CODE QUALITY AND SECURITY REQUIREMENTS

### Rule 6.5: Expert Code Review Protocol
- **Security Analysis**: Mandatory security vulnerability scanning for all code changes;
- **Performance Impact Assessment**: Analyze memory and processing speed implications;
- **Review Quality Gates**: Continue review cycles until no significant concerns remain;

**VALIDATION**: Complete multi-model code review cycle before deployment.

### Rule 6.6: Error-Driven Development
- **Comprehensive Error Analysis**: Capture full error context including stack traces and system state;
- **Proactive Error Handling**: Implement comprehensive try-catch blocks with graceful degradation;
- **User-Friendly Error Messages**: Provide clear error feedback and recovery guidance;

**VALIDATION**: If possible, test error handling scenarios and validate recovery mechanisms.

### Rule 6.7: Real-Time Validation and Testing 
- **Automated Testing Integration**: Run validation tests before code commits;
- **Regression Detection**: Monitor for performance degradation in real-time;
- **Memory Leak Prevention**: Track memory usage patterns continuously;
- **WebSocket Health Monitoring**: Verify connection stability;

**VALIDATION**: If possible, implement automated testing pipeline with real-time monitoring.

---

## TECHNICAL IMPLEMENTATION STANDARDS

### Rule 7: Smart Model Management
- **Startup Optimization**: Download and load only PT-BR language model at startup;
- **Local Caching**: Cache models locally after first download;
- **Storage Efficiency**: Implement efficient model storage management;
- **Model Focus**: Optimize exclusively for PT-BR medium models (openai-whisper, faster-whisper);

**VALIDATION**: If possible, verify model loading efficiency and local caching functionality.

### Rule 8: System Resource Monitoring
- **Resource Usage**: Monitor system power consumption and resource usage. App must be efficient on memory usage;
- **Performance Impact**: Prevent system or browser stuttering and freezing;
- **Parallel Processing**: App must have efficient processing for audio chunks;
- **Accuracy Preservation**: Maintain transcription and diarization accuracy during processing speed optimizations.

**VALIDATION**: If possible, test system resource impact and processing efficiency.

### Rule 9: WebSocket Communication Enhancement
- **Freeze Prevention**: Be aware of WebSocket freezing issues;
- **Progress Updates**: Provide real-time progress updates to users with percentages being shown, keeping user on par of what the app is doing "behind curtains";
- **User Feedback**: Keep users informed about real processing status;
- **UI Responsiveness**: Show path where .srt and .txt were downloaded to, with transcription and diarization previews on interface;
- **Extended Content**: Handle large transcriptions with proper UI scrolling;

**VALIDATION**: If possible, test WebSocket reliability and progress reporting accuracy.

### Rule 10: Implementation Testing Protocol
- **Pre-Deployment Testing**: Test after each implementation;
- **Metrics Verification**: Verify achievement of intended metrics before new changes;
- **Iterative Validation**: Validate reachable performance targets at each step;

**VALIDATION**: If possible, implement mandatory testing checkpoints for Project code changes.

### Rule 10.5: Code Quality Standards
- **Type Checking**: Ensure full type checking and compliance, with proper Unicode handling, including Windows systems;
- **Pylance Compatibility**: Maintain Pylance adequations and standards;
- **Code Reliability**: Implement robust, type-safe code throughout application;

**VALIDATION**: If possible, run type checking validation before code deployment.

---

## COMPREHENSIVE TESTING AND VALIDATION FRAMEWORK

### Rule 10.6: Multi-Step Validation Process 
- **Automated Pre-Validation**: Syntax checking, import validation, and compliance scanning;
- **Integration Testing**: Component interaction testing, WebSocket functionality, file processing validation;
- **Real-World Validation**: Testing with actual audio files in data/recordings folder;
- **Performance Benchmarking**: Processing speed, memory usage, and accuracy testing;

**VALIDATION**: If possible, complete all validation steps before deployment.

### Rule 10.7: Comprehensive Test Suite Integration 
- **Test Consolidation**: All testing consolidated in test_unit.py in c:/transcrevai_windows/tests/ folder;
- **Automated Test Execution**: Pre-commit testing with regression detection;
- **Real-Time Monitoring**: Continuous validation during development;
- **Baseline Establishment**: Record performance metrics for stable versions;

**VALIDATION**: Maintain comprehensive test coverage with automated execution.

---

## HARDWARE AND COMPATIBILITY REQUIREMENTS

### Rule 11: Hardware Optimization
- **Minimum Specifications**: Optimize for 4 physical CPU cores (8 threads), 8GB RAM DDR3, 5GB HDD free space (minimum system requirements);
- **Low-End Hardware**: Ensure functionality on minimum level hardware configurations;
- **Performance Scaling**: Maintain performance targets on minimum viable hardware.

**VALIDATION**: Test matching minimum specifications.

### Rule 12: System Optimization
- **Unicode Cleanup**: Implement proper Unicode handling, including Windows systems, without usage of emoticons on the code or app UI;
- **Memory Limits**: Optimize memory usage for application stability, with efficient memory recycling and process termination after done with usage, avoiding harmful overhead;
- **Efficient Code**: Write memory-efficient code for a stable application;

**VALIDATION**: Test Unicode handling and memory efficiency.

---

## PROJECT ORGANIZATION AND MAINTENANCE

### Rule 13: File Organization Standards
- **Component Separation**: Maintain features in appropriate and corresponding files (audio_processing.py for audio processing related code, diarization.py for diarization related code, transcription.py for transcription related code, etc.);
- **If possible and effective, maintain current project file structure. Ask permission to create or delete files explicitly, with explanations**
- **Code Consistency**: Preserve existing code syntax and flow patterns;
- **Modular Design**: Keep related functionality grouped in corresponding files;

**VALIDATION**: Audit file organization and maintain file structure if possible.

### Rule 14: Documentation and Tracking
- **Documentation Files**: Keep track of system and code modifications through documenting every change made on project in .md files on folder "C:\TranscrevAI_windows\.claude\CHANGES_MADE", to be created specifically for this documenting purpose, named with date and time of file creation (example: 'implementation_DD.MM.YY_HH.MM.SS.md');
- **Conversation History**: Track latest conversations and past implementations;

**VALIDATION**: Adequately keep track of changes made on Project.

### Rule 15: Validation Testing Protocol
- **Critical Testing**: Test application with real audio files in 'c:/TranscrevAI_windows/data/recordings'
- **Expected Results**: Compare obtained results with expected transcription and diarization outputs on respective "expected_results_" file
("filename.wav" with "expected_results_'filename'.txt" - examples:  
"t.speakers.wav" and "expected_results_t.speakers.txt"; 
"t2.speakers.wav" and "expected_results_t2.speakers.txt"; 
"d.speakers.wav" and "expected_results_d.speakers.txt"; 
"q.speakers.wav" with "expected_results_q.speakers.txt");

**VALIDATION**: Mandatory testing with reference audio files and expected output validation.

---

## SYSTEM INTEGRATION REQUIREMENTS

### Rule 16: Application Cohesion
- **Unified Operation**: Ensure all TranscrevAI project files work in unison;
- **Robust Programming**: Maintain accurate and efficient code throughout the app;
- **Testing Consolidation**: Condense all testing into test_unit.py in 'c:/TranscrevAI_windows/tests/' folder, or in specifically created files to be saved on 'c:/TranscrevAI_windows/tests/' folder;
- **Important**: All testing created needs to be implemented (or merged) on single test source file 'c:/transcrevai_windows/tests/test_unit.py', keeping in mind the necessary imports changes and pylance compliance, without errors. File 'c:/transcrevai_windows/tests/conftest.py' has the testing configurations.

**VALIDATION**: Test complete system integration and file organization.

### Rule 17: Storage and Model Optimization
- **Efficient Storage**: Handle model storage efficiently;
- **Single-Model Focus**: Optimize all code exclusively for PT-BR medium models (openai-whisper, faster-whisper);
- **Performance Maximization**: Maximize performance while minimizing memory usage;

**VALIDATION**: Verify storage efficiency and performance optimizations.

### Rule 18: Deployment and Review
- **Docker Packaging**: Package complete TranscrevAI_windows application into Docker container;
- **Review Accessibility**: Enable easy reviewer access to evaluate work;
- **Containerization**: Ensure full application functionality within Docker environment;

**VALIDATION**: Test Docker containerization and deployment process.

---

## FUTURE COMPATIBILITY AND SCALABILITY

### Rule 19: Multi-Platform Foundation
- **Windows Priority**: Perfect Windows performance first;
- **Minimum Hardware**: Ensure flawless operation on 4 CPU cores (8 threads), 8GB RAM DDR3;
- **Future Platforms**: Prepare solid foundation for future Linux, Apple Silicon, NVIDIA, Intel, GPU implementations;
- **Mobile Readiness**: Design architecture compatible with future Android and iOS implementations;
- **Scalable Base**: Build robust foundation that can expand to multiple platforms;

**VALIDATION**: Test on minimum Windows hardware specifications and verify architectural scalability.

---

## ENHANCED COMPLIANCE VALIDATION FRAMEWORK

### Automatic Validation Checkpoints
2. **Performance Metrics**: Processing speed close to 0.75s/1s audio, accuracy >90%
3. **Memory Usage**: masximum RAM consumption 3.5GB during operation
4. **Model Compliance**: Only PT-BR medium models in use
5. **Hardware Compatibility**: Functional on any windows 10/11 machine with 4 cores, 8GB RAM (minimum)
6. **Code Quality**: Type checking, Pylance compliance, and security analysis
7. **Documentation**: Updated "c:/transcrevai_windows/.claude/updates.txt" with latest modifications and brief resume of project's current state
8. **Testing**: Validation against data/recordings/ reference samples with automated test suite
9. **Docker Compatibility**: Successful containerization and deployment
10. **Error Resilience**: Comprehensive error handling and recovery mechanisms

### Enhanced Quality Gates
- **Implementation Planning**: Detailed plan approved by user before coding
- **Incremental Testing**: Each implementation segment tested before proceeding
- **Cross-Model Review**: Both AI models validate approach and implementation
- **Security Scanning**: Vulnerability analysis completed without issues
- **Real-World Testing**: Validation with actual audio files from data/recordings
- **Performance Benchmarking**: All metrics meet or exceed targets
- **Regression Testing**: Existing functionality preserved and enhanced

### Violation Consequences
- **Multi-Model Disagreement**: STOP implementation until consensus achieved
- **Performance Degradation**: Immediate rollback with root cause analysis
- **System Instability**: Revert to last stable implementation with error investigation
- **Security Vulnerabilities**: Fix security issues before proceeding with any changes
- **Resource Overuse**: Memory optimization required with monitoring implementation
- **Compliance Failure**: Re-implementation with strict rule adherence focus

### Success Metrics
- **Speed**: Close as possible, with stability, to 0.75s processing per 1s audio (Target: 100% compliance)
- **Accuracy**: >90% transcription and diarization accuracy (Target: 100% compliance)  
- **Memory**: 3.5GB max. RAM usage (Target: 100% compliance)
- **Stability**: Zero crashes during normal operation by real users (Target: 100% compliance)
- **Hardware**: Functional on minimum specifications (Target: 100% compliance)
- **Security**: No vulnerabilities in code or dependencies (Target: 100% compliance)
- **Testing Coverage**: Comprehensive test suite with automated execution (Target: 100% compliance)

**This enhanced compliance framework ensures systematic validation of all TranscrevAI development against concrete, measurable standards while implementing multi-model validation, comprehensive testing, and advanced quality assurance measures to maintain focus on core performance objectives and system stability.**"