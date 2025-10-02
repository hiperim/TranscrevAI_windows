Simplified project's workflow:

(1)|Audio Input| --> (2)[Platform check]

(2) -->|Desktop| (3/4)[Desktop capture]

(3)/(4) --> (5)[Format validation]

(5) -->|Valid| (6)[.WAV/.MP4 conversion]

(5) -->|Invalid| (7)[Error: reject file]

(6) --> (8)[Parallel processing]

(8) --> (9)[Transcription pipeline]

(9) --> (10)[Diarization pipeline]

(9) --> (11)[Model validation]

(11) -->|Valid| (12)[Chunked automatic speaker recognition (ASR)]

(11) -->|Invalid| (13)[Model download]

(10) --> (14)[Voice activity detection (VAD) filtering]

(14) --> (15)[Speaker clustering]

(12)/(15) --> (16)[Time alignment]

(16) --> (17)[SRT generation]

(17) --> (18)[Output check]

(18) -->|Valid| (19)[MP4 packaging]

(18) -->|Error| (20)[Retry generation]

(19) --> (21)[Final output]
