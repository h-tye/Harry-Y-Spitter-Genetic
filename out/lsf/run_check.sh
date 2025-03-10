cd "$(dirname "$0")" || exit

listOfDirs=$(ls -d ./*/)
for dir in $listOfDirs; do
  numberOfRunLsfFiles=$(ls -1 "./$dir" | grep -c ".*[.]run.*[.]lsf")
  numberOfRunningLsfFiles=$(ls -1 "./$dir" | grep -c ".*[.]running[.]lsf")
  numberOfResults=$(ls -1 "../results/$dir/" | grep -c ".*[.]sqlite")

  dir=${dir%/}
  if [[ $numberOfRunLsfFiles -ne 0 ]]; then
    if [[ $numberOfRunningLsfFiles -ne -1 ]]; then
      echo -e "${numberOfRunLsfFiles} \t ${numberOfResults} \t $((numberOfRunLsfFiles + numberOfResults > 4096 - 1 )) \t $(( numberOfResults * 100 / (numberOfRunLsfFiles + numberOfResults) ))% \t ${dir}"

      if [[ $numberOfRunLsfFiles -lt 100 ]]; then
        limit=$((numberOfRunLsfFiles > 10 ? 10 : numberOfRunLsfFiles))
        for _ in $(seq 1 $limit); do
          sbatch "${dir}/${dir}.lsf.slurm"
        done
      fi
    fi

    continue
  fi

  if [[ $numberOfResults -ne 4096 ]]; then
    echo "!!Number of results: $numberOfResults"
    continue
  fi

  rm -rf "./${dir}/"
  COMPILE_ID=$(sbatch "${dir}.compile.slurm" | awk '{print $4}')
  echo "COMPILE ID: $COMPILE_ID"
  MINIFY_ID=$(sbatch --dependency="afterok:$COMPILE_ID" "${dir}.minify.slurm" | awk '{print $4}')
  echo "MINIFY ID: $MINIFY_ID"
done

listOfSqlite=$(ls -1 ../results/ | grep ".*[.]sqlite")
for sqlite in $listOfSqlite; do
  sqlite=${sqlite%.sqlite}
  if [ ! -d "$sqlite" ]; then
    if [ -f "${sqlite}.compile.slurm" ]; then
      rm "${sqlite}.compile.slurm"
      echo "Removed ${sqlite}.compile.slurm"
    fi
  fi
done

listOfMini=$(ls -1 ../results/ | grep ".*[.]pkl[.]gz")
for mini in $listOfMini; do
  mini=${mini%.pkl.gz}
  if [ ! -d "$mini" ]; then
    if [ -f "${mini}.minify.slurm" ]; then
      rm "${mini}.minify.slurm"
      echo "Removed ${mini}.minify.slurm"
    fi
  fi
done