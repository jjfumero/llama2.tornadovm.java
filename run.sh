#!/bin/bash

./createSPIRVCode.sh

# Print usage information
usage() {
  echo "Usage:"
  echo -e "\tLevel Zero Execution: $0 -v LevelZero <.bin file>"
  echo -e "\tTornadoVM Execution: $0 -v tornadovm <.bin file>"
  echo -e "\tJava Execution: $0 -v java <.bin file>"
  exit 1
}

# Execute Llama2 with LevelZero/TornadoVM/Java
execute_command() {
  if [ "$version" == "java" ]; then
        echo "Running Llama2 Singled-Threaded Java"
        tornado --jvm=" -Dllama2.version=java " -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
  elif [ "$version" == "levelzero" ]; then 
        echo "Running Llama2 with Level Zero JNI and Java"
        tornado --jvm=" -Dllama2.version=levelzero -Dllama2.device=$device" -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
  elif [ "$version" == "tornadovm" ]; then 
        echo "Running Llama2 with TornadoVM"
        tornado --jvm=" -Dllama2.version=tornadovm -Dllama2.device=$device" -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 $token_file
  else
      echo "$version is invalid"
      usage
  fi
}

# Parse command line options to identify arguments for the workgroup size, the vector type, or the java execution mode, if provided
parse_options() {
  device=0
  while getopts ":d:v:j:" opt; do
    case $opt in
      d)
        device="$OPTARG"
        ;;
      v)
        version="$OPTARG"
        ;;
	    j)
	      java="$OPTARG"
	      ;;
      \?)
        echo "Invalid option: -$OPTARG" >&2
        usage
        ;;
      :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        ;;
    esac
  done
}

# Main function
main() {
  # Parse command line options
  parse_options "$@"

  # Shift to get the input token file, which is a mandatory input
  shift $((OPTIND - 1))

  # Check if the token file argument was provided
  if [ $# -eq 0 ]; then
    echo "Error: Missing token file argument." >&2
    usage
  fi

  token_file="$1"

  execute_command
}

main "$@"
