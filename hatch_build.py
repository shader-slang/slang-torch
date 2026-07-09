import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        wheel_tag = os.environ.get("SLANGTORCH_WHEEL_TAG")
        if wheel_tag:
            build_data["tag"] = wheel_tag
            build_data["pure_python"] = False
