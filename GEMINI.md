# Gemini Customization

This file is used to customize Gemini's behavior.

## Instructions

*   **DO NOT** share any confidential information in this file.
*   Use this file to provide instructions and guidelines for Gemini.
*   You can specify coding styles, project conventions, and other preferences.
*   Stay fully compliant with the file compliance.txt, shown below:
    "# TRANSCREVAI PROJECT COMPLIANCE RULES
        *Enhanced for systematic validation and error prevention*

        ###MANDATORY ADHERENCE###: 
        All TranscrevAI development must comply with these rules. Violations will cause system instability and performance degradation;

        Keep track of system and code modifications throught documenting every change made on Project in .md files on folder "C:\TranscrevAI_windows\.claude\CHANGES_MADE" in files to be created specifically for this documenting purpose, named with date and time of file creation (example: 'implementation_DD.MM.YY_HH.MM.SS.md');

        Real audio files for full pipeline processing and testing: 'c:/TranscrevAI_windows/data/recordings';

        TranscrevAI needs to work without crashes or instabilitie em web-browsers, as it functions throught WebSocket.

        ---

        ## CORE PERFORMANCE REQUIREMENTS

        ### Rule 1: Audio Processing Performance Standards
        - **Processing Speed**: Project aspires to reach ~0.75s processing time per 1s of recorded áudio (average between cold and warm starts), with stability to handle web-browser usage (it is a websocket app);
        - **Accuracy Target**: Project aspires to achieve 95%+ accuracy in both transcription and diarization operations;
        - **Language Optimization**: PT-BR (Portuguese Brazilian) exclusive optimization (model and code optimizations);
        - **Model Restriction**: Use "medium" model for all operations (faster-whisper, openai-whisper);
        - **Conversation Focus**: Optimize app and transcription for PT-BR;

        **VALIDATION**: Test all implementations against processing speed and accuracy targets before deployment. On folder "c:/TranscrevAI_windows/data/recordings": for transcription and diarization, compare the real audio files with expected results ("expected_results_'filename'.txt" with "filename.wav" - example:  audio file "t.speakers.wav" and expected results for transcription and diarization "expected_results_t.speakers.txt") for real usage testing validation;


        ### Rule 2: Speaker Diarization Constraints
        - **Dynamic Speaker Detection**: NEVER presume fixed number of speakers;
        - **Adaptive Processing**: Handle variable speaker counts per audio file;
        - **Real-World Optimization**: Optimize for actual usage patterns, not theoretical scenarios;

        **VALIDATION**: On folder "c:/TranscrevAI_windows/data/recordings": for transcription and diarization, compare audio files with expected results ("expected_results_'filename'.txt" with "filename.wav" - example:  audio file "q.speakers.wav" and expected results for transcription and diarization "expected_results_q.speakers.txt") for real usage testing validation;

        ### Rule 3: System Stability Requirements
        - **Historical Context**: Previous over-implementation caused system crashes and instability (be aware);
        - **Incremental Approach**: Implement features gradually with testing at each step if possible;
        - **Performance Maintenance**: Core performance metrics are more important than additional features;
        - **Rollback Capability**: Maintain ability to revert changes that degrade performance, if needed;

        **VALIDATION**: After each implementation, verify system stability and performance metrics.

        ---

        ## RESOURCE OPTIMIZATION REQUIREMENTS

        ### Rule 4: Memory Management
        - **RAM Limit**: Maintain maximum ~3.5gb RAM usage for single PT-BR model operations;
        - **Consider it is a WebSocket app, that will be run on web-browsers like Google Chrome;
        - **Memory Efficiency**: Optimize all code for minimal memory footprint;
        - **Model Loading**: Load only required PT-BR medium model, no multi-model support;

        **VALIDATION**: Monitor memory usage during processing and optimize if too much consumption.

        ### Rule 5: Language and Model Optimization
        - **Exclusive PT-BR Focus**: All improvements and corrections for Portuguese Brazilian only;
        - **Model Strategy**: Use exclusively "medium" model for Portuguese Brazilian;
        - **No Multi-Language Support**: Remove or disable any multi-language capabilities;

        **VALIDATION**: Audit codebase for PT-BR exclusive optimization.

        ### Rule 6: Performance Optimization Strategy
        - **Efficiency First**: Prioritize efficient and beneficial optimizations, alongside with accurate transcription and diarization;
        - **Application Intent**: Align all optimizations with TranscrevAI core intentions and aspirations;
        - **Measurable Improvements**: Implement only optimizations with quantifiable benefits;

        **VALIDATION**: Measure performance impact of all optimizations with concrete metrics and results to be displayed and analyzed.

        ---

        ## TECHNICAL IMPLEMENTATION STANDARDS

        ### Rule 7: Smart Model Management
        - **Startup Optimization**: Download and load only PT-BR language model at startup;
        - **Local Caching**: Cache models locally after first download;
        - **Storage Efficiency**: Implement efficient model storage management;
        - **Model Focus**: Optimize exclusively for PT-BR medium models (openai-whisper, faster-whisper);

        **VALIDATION**: Verify model loading efficiency and local caching functionality.

        ### Rule 8: System Resource Monitoring
        - **Resource Usage**: Monitor system power consumption and resource usage. App must be efficient on memory usage;
        - **Performance Impact**: Prevent system or browser stuttering and freezing;
        - **Parallel Processing**: Implement efficient parallel processing for audio chunks;
        - **Accuracy Preservation**: Maintain transcription and diarization accuracy during processing speed optimizations.

        **VALIDATION**: Test system resource impact and parallel processing efficiency.

        ### Rule 9: WebSocket Communication Enhancement
        - **Freeze Prevention**: Implement WebSocket freezing fixes;
        - **Progress Updates**: Provide real-time progress updates to users with percentages being shown, keeping user on par of what the app is doing "behind curtains";
        - **User Feedback**: Keep users informed about real processing status;
        - **UI Responsiveness**: Show transcription and diarization results with scrollable interface;
        - **Extended Content**: Handle large transcriptions with proper UI scrolling;

        **VALIDATION**: Test WebSocket reliability and progress reporting accuracy.

        ### Rule 10: Implementation Testing Protocol
        - **Pre-Deployment Testing**: Test after each implementation;
        - **Metrics Verification**: Verify achievement of intended metrics before new changes;
        - **Iterative Validation**: Validate reachable performance targets at each step;

        **VALIDATION**: Implement mandatory testing checkpoints for Project code changes.

        ### Rule 10.5: Code Quality Standards
        - **Type Checking**: Ensure full type checking and compliance, with proper Unicode handling on Windows systems;
        - **Pylance Compatibility**: Maintain Pylance adequations and standards;
        - **Code Reliability**: Implement robust, type-safe code throughout application;

        **VALIDATION**: Run type checking validation before code deployment.

        ---

        ## HARDWARE AND COMPATIBILITY REQUIREMENTS

        ### Rule 11: Hardware Optimization
        - **Minimum Specifications**: Optimize for 4 physical CPU cores (8 threads), 8GB RAM DDR3, 5GB HDD free space (minimum system requirements);
        - **Low-End Hardware**: Ensure functionality on minimum level hardware configurations;
        - **Performance Scaling**: Maintain performance targets on minimum viable hardware.

        **VALIDATION**: Test matching minimum specifications.

        ### Rule 12: System Optimization
        - **Unicode Cleanup**: Implement proper Unicode handling for Windows systems, without usage of emoticons on the code or app UI;
        - **Memory Limits**: Optimize memory usage for application stability, with efficient memory recycling and process termination after done with usage, avoiding harmful overhead;
        - **Efficient Code**: Write memory-efficient code for a stable application;

        **VALIDATION**: Test Unicode handling and memory efficiency on Windows systems.

        ---

        ## PROJECT ORGANIZATION AND MAINTENANCE

        ### Rule 13: File Organization Standards
        - **Component Separation**: Maintain features in appropriate and corresponding files (audio_processing.py for audio processing related code, diarization.py for diarization related code, transcription.py for transcription related code, etc.);
        - **Code Consistency**: Preserve existing code syntax and flow patterns;
        - **Modular Design**: Keep related functionality grouped in corresponding files;

        **VALIDATION**: Audit file organization and maintain file structure if possible.

        ### Rule 14: Documentation and Tracking
        - **Documentation Files**: Keep track of system and code modifications throught documenting every change made on Project in .md files on folder "C:\TranscrevAI_windows\.claude\CHANGES_MADE" in files to be created specifically for this documenting purpose, named with date and time of file creation (example: 'implementation_DD.MM.YY_HH.MM.SS.md');
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
        - **Important**: All testing created needs to be implemented (or merged) on single test source file 'c:/transcrevai_windows/tests/test_unit.py', keeping in mind the necessry imports changes and pylance compliance, without errors. File 'c:/transcrevai_windows/tests/conftest.py' has the testing configurations.

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
        - **Windows Priority**: Perfect Windows notebook performance first;
        - **Minimum Hardware**: Ensure flawless operation on 4 CPU cores (8 threads), 8GB RAM DDR3;
        - **Future Platforms**: Prepare solid foundation for future Linux, Apple Silicon, NVIDIA, Intel GPU implementations;
        - **Mobile Readiness**: Design architecture compatible with future Android and iOS implementations;
        - **Scalable Base**: Build robust foundation that can expand to multiple platforms;

        **VALIDATION**: Test on minimum Windows hardware specifications and verify architectural scalability.

        ---

        ## COMPLIANCE VALIDATION FRAMEWORK

        ### Automatic Validation Checkpoints
        1. **Performance Metrics**: Processing speed close to 0.5s/1s audio, accuracy >95%
        2. **Memory Usage**: RAM consumption close to 2GB during operation
        3. **Model Compliance**: Only PT-BR medium model in use
        4. **Hardware Compatibility**: Functional on any windows 10/11 machine with 4 cores, 8GB RAM (minimum)
        5. **Code Quality**: Type checking and Pylance compliance
        6. **Documentation**: Updated "c:/transcrevai_windows/.claude/fixes.txt" with latest fixes and brief resume of project's current state 
        7. **Testing**: Validation against data/recordings/ reference samples
        8. **Docker Compatibility**: Successful containerization and deployment

        ### Violation Consequences
        - **Performance Degradation**: Immediate rollback of changes
        - **System Instability**: Revert to last stable implementation
        - **Resource Overuse**: Memory optimization required before proceeding
        - **Compliance Failure**: Re-implementation with rule adherence focus

        ### Success Metrics
        - **Speed**: close as possible, with stability, to 0.5s processing per 1s audio (Target: 100% compliance)
        - **Accuracy**: >90% transcription and diarization accuracy (Target: 100% compliance)  
        - **Memory**: close as possible, with stability, to 2GB RAM usage (Target: 100% compliance)
        - **Stability**: Zero crashes during normal operation by real users (Target: 100% compliance)
        - **Hardware**: Functional on minimum specifications (Target: 100% compliance)

        **This compliance framework ensures systematic validation of all TranscrevAI development against concrete, measurable standards while maintaining focus on core performance objectives.**"

