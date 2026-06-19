# Performance & Benchmarks

The transition from an Object-Oriented Programming (OOP) architecture to a Data-Oriented Design (DOD) was driven by strict physical constraints encountered during the performance testing phase.

## OOP Architecture Test Results

**Initial OOP Implementation (Low Scale):**
| Agents | Ticks | Total Time (s) | Throughput (Rows/s) | Peak RAM (MB) |
|--------|-------|----------------|----------------------|----------------|
| 100    | 100   | 22.33s         | 242.0                | 292.74         |
| 1,000  | 100   | 62.78s         | 760.0                | 234.62         |
| 5,000  | 200   | 307.41s        | 732.0                | 243.93         |
| 15,000 | 200   | 904.92s        | 678.0                | 296.55         |
| 25,000 | 300   | 1512.07s       | 657.0                | 372.40         |

**Extended OOP Implementation (Stress Test):**
| Agents | Radius | Ticks | Elapsed (s) | Rows Generated | Throughput (Rows/s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 30,000 | 1500m | 100 | 258.81s | 1,445,833 | 5,586 |
| 50,000 | 2000m | 100 | 539.71s | 2,796,979 | 5,182 |
| 75,000 | 3000m | 150 | 1317.36s | 5,663,755 | 4,299 |
| 100,000| 5000m | 200 | **SYSTEM CRASH** | -- | -- |

The primary flaw in the Python OOP approach is **memory fragmentation**. Every `Agent` instance is an object allocated at an arbitrary memory address. When the CPU processes a list of these objects, it must continuously retrieve pointers to distant memory locations. This prevents the processor from effectively utilizing the cache, forcing continuous and slow RAM access. Furthermore, the OOP model requires the CPU to jump between objects in memory, drastically throttling throughput.

Additionally, the simulation lifecycle was initially managed via a standard loop (e.g., `for agent in agents: agent.step()`). The underlying issue is that for every single iteration, the interpreter must perform a dictionary lookup to find the `step` method for the instance and its class. Because Python is a dynamically typed language, it constantly validates whether the object is correct and authorized to execute that command, creating massive execution overhead.

Finally, a Python object carries a substantial amount of metadata alongside its core data, including internal dictionaries, Garbage Collector pointers, and type information. Even in the 25,000-agent test, this bloat resulted in a memory footprint of 372 MB, inevitably leading to an *Out of Memory* crash when approaching the 100,000-agent threshold.


## The Shift to a DOD Architecture

Overcoming the structural limits of the OOP model was achieved by pivoting to a **Data-Oriented** paradigm. In this architecture, we no longer process "Agent" entities; instead, we manipulate raw data organized in contiguous memory structures.

**Vectorization:** The core innovation lies in the elimination of Python `for` loops in favor of vectorization. By replacing class instances with NumPy matrices, computation is delegated to underlying **C** functions, which are highly optimized for modern CPU architectures.
Agent data is stored in sequential memory blocks. This allows the CPU to fully leverage the **L1/L2 Cache**, eliminating the idle times caused by OOP cache misses. Moreover, the processor can now apply the same instruction to multiple data points simultaneously using **SIMD (Single Instruction, Multiple Data)**, computing dozens of spatial displacements in a single clock cycle.

### The I/O Bottleneck
Testing the new vectorized engine revealed a new challenge: the massive disparity between pure calculation speed and data *serialization* speed. 
The pure engine, operating exclusively on RAM matrices, reaches peaks of **150,000 agents per second**. In this state, the CPU is completely saturated by SIMD mathematical operations. However, when the system must communicate externally (I/O phase), it must convert these tensors into readable formats, such as JSON, for the API. This serialization reduces the throughput to approximately 60,000 agents/s. 

To address this, the `/step` API utilizes a parameter allowing the client to specify whether it needs the payload returned after the iteration. This empowers the client to choose between receiving real-time positional evolution (high I/O) or simply retrieving the data only at the end of the simulation (zero I/O overhead).

Below is the comparison between the two modes:

**Full I/O (Data Serialized Every Tick):**
| Agents | Ticks | Total Time (s) | Throughput (Rows/s) | Peak RAM (MB) |
|--------|-------|----------------|----------------------|----------------|
| 100    | 100   | 0.37s          | 26,926               | 68.96          |
| 1,000  | 100   | 1.79s          | 55,653               | 69.42          |
| 5,000  | 200   | 16.36s         | 61,121               | 72.29          |
| 15,000 | 200   | 49.23s         | 60,938               | 77.73          |
| 25,000 | 300   | 119.69s        | 62,658               | 82.75          |

**PURE ENGINE (Zero I/O Overhead):**
| Agents | Ticks | Total Time (s) | Throughput (Rows/s) | Peak RAM (MB) |
|--------|-------|----------------|----------------------|----------------|
| 100    | 100   | 0.25s          | 38,774               | 68.96          |
| 1,000  | 100   | 0.77s          | 128,633              | 69.42          |
| 5,000  | 200   | 6.67s          | 149,925              | 72.29          |
| 15,000 | 200   | 20.27s         | 148,044              | 77.73          |
| 25,000 | 300   | 50.09s         | 149,718              | 82.75          |


### Parallel Population
Despite the efficiency gained in movement simulation, the initialization phase (calculating paths via Dijkstra's algorithm) remained the highest computational cost. In Python, the Global Interpreter Lock (GIL) prevents threads from executing CPU-intensive calculations in parallel, making the generation of 100,000 routes an extremely slow, sequential process.

To bypass this limitation, the system implements a **Parallel Population** architecture based on *Multiprocessing*. Unlike threads, multiprocessing spawns independent OS processes, each with its own interpreter instance and memory space. This allows the software to genuinely utilize all CPU cores simultaneously. 
The total population is divided into uniform "batches" distributed across the available cores. Each core autonomously calculates the shortest paths for its specific agent subset. Once the parallel calculations conclude, the raw results are collected and packed directly into the pre-allocated NumPy matrices.

Thanks to this architecture, population throughput remains stable as the load increases. Without this approach, the waiting time to populate metropolitan-scale scenarios would have been prohibitive, frequently causing system crashes due to API timeouts.

**Population Test (Without Parallelization):**
| Agents  | Population Time (s) | Simulation Throughput (Agents/s) | Simulation Latency |
|---------|---------------------|------------------------------------|---------------------|
| 50,000  | 2512.56s * | 68,196                             | 0.733s              |
| 100,000 | 5263.15s * | 96,124                             | 1.040s              |

**Population Test (With Parallelization):**
| Agents  | Population Time (s) | Simulation Throughput (Agents/s) | Simulation Latency |
|---------|---------------------|------------------------------------|---------------------|
| 50,000  | 1504.39s            | 149,925                            | 0.333s              |
| 100,000 | 3255.46s            | 150,000                            | 0.666s              |


## Physical Breaking Point
To validate the ultimate scalability of the DOD architecture, a stress test was conducted to identify the physical *Breaking Point* of the system on a workstation equipped with **32 GB of RAM**. 
To isolate the calculation engine's performance from the I/O times caused by Dijkstra pre-processing, an alternative instantaneous population technique was implemented. This allows the immediate allocation of vectorized data structures in memory, enabling the evaluation of the system's behavior with a massive number of active entities.

**Stress Test Results:**
| Total Agents | Latency per Tick | Throughput (Rows/s) | RAM Occupancy | System Status |
|--------------|------------------|----------------------|----------------|---------------|
| 100,000      | 0.38s            | 263,157              | 34.4%          | PASSED        |
| 500,000      | 1.81s            | 276,243              | 44.2%          | PASSED        |
| 1,000,000    | 3.62s            | 276,243              | 55.8%          | PASSED        |
| 1,500,000    | 5.61s            | 267,379              | 67.4%          | PASSED        |
| 2,000,000    | 7.51s            | 266,311              | 77.6%          | PASSED        |
| 2,500,000    | --               | --                   | --             | MEMORY ERROR  |

The results clearly demonstrate that in the new simulation engine, the bottleneck is no longer execution speed (as in the OOP model), but strictly the physical capacity of the machine's memory. 
Up to the **2,000,000 agent** threshold, the system exhibits near-linear scalability, maintaining a constant throughput above 260,000 processed rows per second. This proves the architecture is extremely efficient at managing resources, with computation time growing predictably relative to the data volume.

The failure observed at 2,500,000 agents is caused by a physical RAM addressing limitation. In a NumPy-based DOD architecture, data must be stored in contiguous memory blocks to permit the CPU to execute SIMD instructions. To handle 2.5 million agents, the system attempted to allocate the following simultaneously:

* *node_path_matrix* (int64): ~9.31 GiB
* *path_matrix* (float32): ~9.31 GiB
* State structures and overhead: ~2.5 GiB
    
Although the workstation has 32 GB of RAM, the Operating System was unable to find a single, contiguous address block large enough to house these matrices, due to the natural RAM fragmentation caused by background processes. The *Memory Error* is, therefore, a safety intervention by the OS to prevent a total hardware lockup (kernel panic).

The fact that the latency per tick remains proportionally constant up to the RAM limit suggests the architecture is fully exploiting the memory hierarchy. By organizing data sequentially, we minimized cache misses; the CPU almost always finds the data it needs in its fast registers (L1/L2), without waiting for the significantly slower response times of the main RAM.

## Conclusions
The transition to the Data-Oriented architecture successfully allowed the system to:

* **Increase the simulation ceiling** from 75,000 to 2,000,000 agents on a standard 32GB RAM workstation.
* **Guarantee deterministic execution**, ensuring the time required to process a tick is strictly linear with respect to the data volume.
* **Exploit SIMD (Single Instruction, Multiple Data) processor instructions**, computing dozens of spatial coordinate updates in a single clock cycle.