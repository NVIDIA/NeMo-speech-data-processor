import re
import portion as P
from packaging.version import Version
from typing import List, Dict
import subprocess

from sdp.logging import logger

MIN_VERSION = Version("0.0.0")
MAX_VERSION = Version("999999.999999.999999")

def parse_constraint(s: str):
    pattern = r"^(>=|<=|==|!=|<|>)\s*(.+)$"
    match = re.fullmatch(pattern, s.strip())
    if not match:
        raise ValueError(f"Invalid constraint: {s}")
    op, version_str = match.groups()
    version = Version(version_str)

    dispatch = {
        ">":    lambda: P.open(version, MAX_VERSION),
        ">=":   lambda: P.closedopen(version, MAX_VERSION),
        "<":    lambda: P.open(MIN_VERSION, version),
        "<=":   lambda: P.openclosed(MIN_VERSION, version),
        "==":   lambda: P.closed(version, version),
        "!=":   lambda: P.open(MIN_VERSION, version) | P.open(version, MAX_VERSION),
    }

    return dispatch[op]()

def resolve_version(constraints: List[str]):
    if len(constraints) == 0:
        raise ValueError()
    
    intersection = parse_constraint(constraints[0])

    if len(constraints) > 1:
        for constraint in constraints[1:]:
            version_range = parse_constraint(constraint)
            intersection &= version_range
    
    interval = intersection._intervals[-1]

    if len(interval) == 0:
        return None
    
    if (interval.lower == interval.upper and 
        interval.left == P.CLOSED and
        inerval.right == P.CLOSED):
        return f"=={interval.lower}" 
    
    compatible = []
    if interval.lower != MIN_VERSION:
        if interval.left == P.CLOSED:
            compatible.append(f">={interval.lower}")
        else:
            compatible.append(f">{interval.lower}")

    if interval.upper != MAX_VERSION:
        if interval.right == P.CLOSED:
            compatible.append(f"<={interval.upper}")
        else:
            compatible.append(f"<{interval.upper}")
    
    return ",".join(compatible)


def merge_requirements(*requirements):
    merged_pkg = dict()
    for _requirements in requirements:
        for pkg, ver in _requirements.items():
            if pkg not in merged_pkg:                
                merged_pkg[pkg] = set()
            
            merged_pkg[pkg].update(ver)
    
    resolved_pkg = dict()
    for pkg, ver in merged_pkg.items():
        if len(ver) == 1 and ver[0] == "":
            resolved_ver = ""
        else:
            resolved_ver = resolve_version(list(ver))
        
            if resolved is None:
                err = [f"'{pkg}{v}'" for v in ver]
                raise ValueError(f"Version conflict: '{', '.join(err)}.")

        resolved_pkg[pkg] = resolved_ver
    return resolved_pkg


def install_packages(packages: List[str]):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {packages}")
        print(f"Error code: {e.returncode}")
        raise


def setup(*requirements):
    packages_to_install = merge_requirements(*requirements)
    
    if len(packages_to_install) > 0:
        logger.info(f'The following packages will be installed:')
        for pkg in packages_to_install: 
            logger.info(f"{pkg.ljust(max_len)}  {ver}")
        
        pip_packages = [f'{pkg}{ver}' for pkg, ver in packages_to_install.items()]
        install_packages(pip_packages)