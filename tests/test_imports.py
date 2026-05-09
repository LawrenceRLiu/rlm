"""Tests to verify all imports are correct and non-conflicting."""

import importlib
import sys
from collections import defaultdict

import pytest


class TestTopLevelImports:
    """Test top-level package imports."""

    def test_rlm_import(self):
        import rlm

        assert hasattr(rlm, "RLM")
        assert "RLM" in rlm.__all__

    def test_rlm_rlm_import(self):
        from rlm import RLM

        assert RLM is not None

    def test_rlm_core_rlm_import(self):
        from rlm.core.rlm import RLM

        assert RLM is not None


class TestClientImports:
    """Test client module imports."""

    def test_clients_module_import(self):
        import rlm.clients

        assert hasattr(rlm.clients, "get_client")
        assert hasattr(rlm.clients, "BaseLM")

    def test_base_lm_import(self):
        from rlm.clients.base_lm import BaseLM

        assert BaseLM is not None

    def test_openai_client_import(self):
        pytest.importorskip("openai")
        from rlm.clients.openai import OpenAIClient

        assert OpenAIClient is not None

    def test_anthropic_client_import(self):
        pytest.importorskip("anthropic")
        from rlm.clients.anthropic import AnthropicClient

        assert AnthropicClient is not None

    def test_portkey_client_import(self):
        pytest.importorskip("portkey_ai")
        from rlm.clients.portkey import PortkeyClient

        assert PortkeyClient is not None

    def test_get_client_function(self):
        from rlm.clients import get_client

        assert callable(get_client)


class TestCoreImports:
    """Test core module imports."""

    def test_core_types_import(self):
        from rlm.core.types import (
            ClientBackend,
            ModelUsageSummary,
            QueryMetadata,
            RLMMetadata,
            UsageSummary,
            WorkspaceAction,
            WorkspaceIteration,
            WorkspaceObservation,
            WorkspaceSnapshot,
        )

        assert ClientBackend is not None
        assert ModelUsageSummary is not None
        assert QueryMetadata is not None
        assert RLMMetadata is not None
        assert UsageSummary is not None
        assert WorkspaceAction is not None
        assert WorkspaceIteration is not None
        assert WorkspaceObservation is not None
        assert WorkspaceSnapshot is not None

    def test_core_rlm_import(self):
        from rlm.core.rlm import RLM

        assert RLM is not None

    def test_core_lm_handler_import(self):
        from rlm.core.lm_handler import LMHandler

        assert LMHandler is not None

    def test_core_comms_utils_import(self):
        from rlm.core.comms_utils import (
            LMRequest,
            LMResponse,
            send_lm_request,
            send_lm_request_batched,
            socket_recv,
            socket_send,
        )

        assert LMRequest is not None
        assert LMResponse is not None
        assert callable(send_lm_request)
        assert callable(send_lm_request_batched)
        assert callable(socket_recv)
        assert callable(socket_send)

    def test_core_config_import(self):
        from rlm.core.config import (
            DockerConfig,
            ObservationConfig,
            ParseConfig,
            RecursionConfig,
            WorkspaceConfig,
        )

        assert DockerConfig is not None
        assert ObservationConfig is not None
        assert ParseConfig is not None
        assert RecursionConfig is not None
        assert WorkspaceConfig is not None


class TestEnvironmentImports:
    """Test environment module imports."""

    def test_environments_module_import(self):
        import rlm.environments

        assert hasattr(rlm.environments, "BaseWorkspaceEnv")
        assert hasattr(rlm.environments, "DockerWorkspaceEnv")

    def test_base_workspace_import(self):
        from rlm.environments.base_workspace import BaseWorkspaceEnv

        assert BaseWorkspaceEnv is not None

    def test_docker_workspace_import(self):
        from rlm.environments.docker_workspace import DockerWorkspaceEnv

        assert DockerWorkspaceEnv is not None


class TestLoggerImports:
    """Test logger module imports."""

    def test_logger_module_import(self):
        import rlm.logger

        assert hasattr(rlm.logger, "RLMLogger")
        assert hasattr(rlm.logger, "VerbosePrinter")
        assert "RLMLogger" in rlm.logger.__all__
        assert "VerbosePrinter" in rlm.logger.__all__

    def test_rlm_logger_import(self):
        from rlm.logger.rlm_logger import RLMLogger

        assert RLMLogger is not None

    def test_verbose_import(self):
        from rlm.logger.verbose import VerbosePrinter

        assert VerbosePrinter is not None


class TestUtilsImports:
    """Test utils module imports."""

    def test_action_parser_import(self):
        from rlm.utils.action_parser import ActionSchema, default_schemas, parse

        assert ActionSchema is not None
        assert callable(default_schemas)
        assert callable(parse)

    def test_prompts_import(self):
        from rlm.utils.prompts import (
            WORKSPACE_SYSTEM_PROMPT_TEMPLATE,
            build_parse_retry_message,
            build_workspace_initial_user_prompt,
            build_workspace_system_prompt,
            format_workspace_iteration,
            render_observation,
        )

        assert WORKSPACE_SYSTEM_PROMPT_TEMPLATE is not None
        assert callable(build_parse_retry_message)
        assert callable(build_workspace_initial_user_prompt)
        assert callable(build_workspace_system_prompt)
        assert callable(format_workspace_iteration)
        assert callable(render_observation)

    def test_provenance_import(self):
        from rlm.utils.provenance import ProvenanceStore

        assert ProvenanceStore is not None

    def test_rlm_utils_import(self):
        from rlm.utils.rlm_utils import filter_sensitive_keys

        assert callable(filter_sensitive_keys)


class TestImportConflicts:
    """Test for import conflicts and naming issues."""

    def test_no_duplicate_names_in_rlm_all(self):
        import rlm

        if hasattr(rlm, "__all__"):
            all_items = rlm.__all__
            assert len(all_items) == len(set(all_items)), (
                f"Duplicate items in rlm.__all__: {all_items}"
            )

    def test_no_duplicate_names_in_logger_all(self):
        import rlm.logger

        if hasattr(rlm.logger, "__all__"):
            all_items = rlm.logger.__all__
            assert len(all_items) == len(set(all_items)), (
                f"Duplicate items in rlm.logger.__all__: {all_items}"
            )

    def test_all_declarations_match_exports(self):
        import rlm
        import rlm.logger

        if hasattr(rlm, "__all__"):
            for name in rlm.__all__:
                assert hasattr(rlm, name), f"rlm.__all__ declares '{name}' but it's not exported"

        if hasattr(rlm.logger, "__all__"):
            for name in rlm.logger.__all__:
                assert hasattr(rlm.logger, name), (
                    f"rlm.logger.__all__ declares '{name}' but it's not exported"
                )

    def test_no_circular_imports(self):
        core_modules = [
            "rlm",
            "rlm.clients",
            "rlm.clients.base_lm",
            "rlm.core",
            "rlm.core.types",
            "rlm.core.config",
            "rlm.core.rlm",
            "rlm.core.lm_handler",
            "rlm.core.comms_utils",
            "rlm.environments",
            "rlm.environments.base_workspace",
            "rlm.environments.docker_workspace",
            "rlm.logger",
            "rlm.logger.rlm_logger",
            "rlm.logger.verbose",
            "rlm.utils",
            "rlm.utils.action_parser",
            "rlm.utils.prompts",
            "rlm.utils.provenance",
            "rlm.utils.rlm_utils",
        ]

        optional_modules = [
            ("rlm.clients.openai", "openai"),
            ("rlm.clients.anthropic", "anthropic"),
            ("rlm.clients.portkey", "portkey_ai"),
        ]

        for module_name in core_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

        for module_name, dependency in optional_modules:
            try:
                importlib.import_module(dependency)
            except ImportError:
                continue

            if module_name in sys.modules:
                del sys.modules[module_name]
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_no_naming_conflicts_across_modules(self):
        module_exports: dict[str, set[str]] = {}

        import rlm
        import rlm.clients
        import rlm.environments
        import rlm.logger

        if hasattr(rlm, "__all__"):
            module_exports["rlm"] = set(rlm.__all__)
        else:
            module_exports["rlm"] = {name for name in dir(rlm) if not name.startswith("_")}

        if hasattr(rlm.clients, "__all__"):
            module_exports["rlm.clients"] = set(rlm.clients.__all__)
        else:
            module_exports["rlm.clients"] = {
                name for name in dir(rlm.clients) if not name.startswith("_")
            }

        if hasattr(rlm.environments, "__all__"):
            module_exports["rlm.environments"] = set(rlm.environments.__all__)
        else:
            module_exports["rlm.environments"] = {
                name for name in dir(rlm.environments) if not name.startswith("_")
            }

        if hasattr(rlm.logger, "__all__"):
            module_exports["rlm.logger"] = set(rlm.logger.__all__)
        else:
            module_exports["rlm.logger"] = {
                name for name in dir(rlm.logger) if not name.startswith("_")
            }

        name_to_modules: dict[str, list[str]] = defaultdict(list)
        for module_name, exports in module_exports.items():
            for export_name in exports:
                name_to_modules[export_name].append(module_name)

        conflicts = {name: modules for name, modules in name_to_modules.items() if len(modules) > 1}
        expected_duplicates = {
            "__file__",
            "__name__",
            "__package__",
            "__path__",
            "__doc__",
            "__loader__",
            "__spec__",
            "__cached__",
            "Any",
            "Literal",
            "Optional",
            "Union",
            "Dict",
            "List",
            "Tuple",
            "Callable",
        }
        conflicts = {
            name: modules for name, modules in conflicts.items() if name not in expected_duplicates
        }

        if conflicts:
            conflict_msg = "\n".join(
                f"  '{name}' exported from: {', '.join(modules)}"
                for name, modules in conflicts.items()
            )
            pytest.fail(f"Found naming conflicts across modules:\n{conflict_msg}")


class TestImportCompleteness:
    """Test that all expected imports are available."""

    def test_all_client_classes_importable(self):
        from rlm.clients.base_lm import BaseLM

        assert isinstance(BaseLM, type)

        try:
            pytest.importorskip("openai")
            from rlm.clients.openai import OpenAIClient

            assert isinstance(OpenAIClient, type)
        except Exception:
            pass

        try:
            pytest.importorskip("anthropic")
            from rlm.clients.anthropic import AnthropicClient

            assert isinstance(AnthropicClient, type)
        except Exception:
            pass

        try:
            pytest.importorskip("portkey_ai")
            from rlm.clients.portkey import PortkeyClient

            assert isinstance(PortkeyClient, type)
        except Exception:
            pass

    def test_all_environment_classes_importable(self):
        from rlm.environments.base_workspace import BaseWorkspaceEnv
        from rlm.environments.docker_workspace import DockerWorkspaceEnv

        assert isinstance(BaseWorkspaceEnv, type)
        assert isinstance(DockerWorkspaceEnv, type)
