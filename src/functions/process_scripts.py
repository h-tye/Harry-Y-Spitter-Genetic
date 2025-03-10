from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Union, Dict, Tuple, Optional

from out.results import get_results_path
from src.functions.__const__ import HASH_LENGTH
from src.lsf_scripts import get_lsf_scripts_path

END_SCRIPT = r'''
    #file name formatting
    filename = currentscriptname;
    #must be whatever filename starts with
    script_idx = length(pwd) + 2;
    filename = substring(filename, script_idx);
    lms_file = replacestring(filename, ".lsf",".lms");

    #save configuartion to text file
    txt_file = replacestring(filename, ".lsf", ".txt");
    write(txt_file, holeMatrix, "overwrite");

    print(lms_file);
    save(lms_file);
    #run;
    save(lms_file);
'''


def get_relative_path(path1: Union[str, Path], path2: Union[str, Path]) -> str:
    path1 = Path(path1).expanduser().absolute()
    path2 = Path(path2).expanduser().absolute()
    common_path = Path(os.path.commonpath([path1, path2]))
    path1_to_common = path1.relative_to(common_path)
    path2_to_common = path2.relative_to(common_path)
    path1_to_common = Path("/".join([".."] * len(path1_to_common.parts)))
    path2_to_common = Path("/".join(path2_to_common.parts))
    return str(path1_to_common / path2_to_common).replace("\\", "/")


def process_scripts(
        parameters: Dict[str, float],
        location: Union[Path, str],
        setup_script: str,
        script_name: str,
        index: Optional[Union[str, int]] = None,
) -> Tuple[str, str]:
    #parameters = [(k, v if isinstance(v, str) else f"{v:.12f}") for k, v in parameters.items()]
    #parameters = {k: v for k, v in parameters if v is not None}
    point_str = sha256(json.dumps(parameters, sort_keys=True).encode("utf-8")).hexdigest()[:HASH_LENGTH]
    if index is None:
        name = f'{script_name}_{point_str}'
    else:
        name = f'{script_name}_{index}_{point_str}'

    scripts = [setup_script, END_SCRIPT]

    for i in range(len(scripts)):
        scripts[i] = scripts[i].replace("{script_name}", script_name)
        scripts[i] = scripts[i].replace("{lsf_scripts_path}", get_relative_path(location, get_lsf_scripts_path()))
        scripts[i] = scripts[i].replace("{results_path}", get_relative_path(location, get_results_path() / script_name))
        scripts[i] = scripts[i].replace("{name}", name)
        # for k, v in parameters.items():
        #     scripts[i] = scripts[i].replace(f"{{{k}}}", v)

    return name, "\n".join(scripts)
