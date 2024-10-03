# An extension of the Llama2.java implementation, accelerated with GPUs by using TornadoVM and Level Zero JNI (GPUs)

<img src="https://github.com/mikepapadim/llama2.tornadovm.java/assets/8652854/4493fe14-7427-4532-91fa-7299cd96034b" width="30%">

This repository provides an implementation of [llama2.java](https://github.com/mukel/llama2.java), extended to use the Vector API and [TornadoVM](https://github.com/beehive-lab/TornadoVM) for acceleration.

## Prerequisites
* **JDK 21+**: This is essential as the project uses the [Project Panama](https://openjdk.org/projects/panama/) for native memory allocation. 
* **TornadoVM**: Detailed installation instructions can be found [here](https://tornadovm.readthedocs.io/en/latest/installation.html).  

## Build
The `set_paths.sh` file provides a template with all the paths that need to be set up for the compilation and execution.
From this template, the paths that need to be set are: 
* **$JAVA_HOME**, with the path to JDK 21
* **$TORNADO_ROOT**, with the path to the TornadoVM installation  

```bash
mvn clean package
```

## Execution
### Token files
Just like the original Java implementation, the program requires a `tokenizer.bin` file and the input models available in the TinyLlamas. 
```bash
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```
### How to run
The repository contains a `run.sh` script for running. This script takes the following arguments:
* Version to run ( `java`, `levelzero`, `tornadovm` )
* Device index to run ( `java`, `levelzero`, `tornadovm` )
* The .bin model file

```bash
// Run with just the model with LevelZero
./run.sh -v levelzero stories15M.bin 
// Run in pure Java, without TornadoVM
./run.sh -v java stories15M.bin 
// Run with TornadoVM
./run.sh -v tornadovm stories15M.bin 

## Change device
// Run with just the model with LevelZero and Device 1
./run.sh -v levelzero -d 1 stories15M.bin 

// Run with TornadoVM and device 1
./run.sh -v tornadovm -d 1 stories15M.bin 
```

## License

MIT
