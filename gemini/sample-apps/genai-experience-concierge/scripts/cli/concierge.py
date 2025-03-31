# Copyright 2025 Google. This software is provided as-is, without warranty or
# representation for any use or purpose. Your use of it is subject to your
# agreement with Google.

import click
import yaml

from scripts.cli import langgraph_demo


@click.group(help="GenAI Experience Concierge demo tool.")
@click.option(
    "-f",
    "--config-file",
    required=False,
    help="YAML config file to configure command/subcommand defaults.",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    default=None,
)
@click.pass_context
def concierge(ctx: click.Context, config_file: str | None = None):
    ctx.ensure_object(dict)

    if config_file is None:
        return

    with open(config_file, "r") as f:
        default_map = yaml.safe_load(f)
        ctx.default_map = default_map


def langgraph_demo_group():
    pass


langgraph_group = concierge.group(
    name="langgraph",
    help="GenAI Experience Concierge demo orchestrated with LangGraph.",
)(langgraph_demo_group)

langgraph_dataset_creation_cmd = langgraph_group.command(
    help="Create a Cymbal Retail dataset and embedding model in the target project. NOTE: This command does not need to be run if using the end-to-end deployment."
)(langgraph_demo.create_dataset)

langgraph_deploy_cmd = langgraph_group.command(
    help="End-to-end deployment including project creation, infrastructure provisioning, container builds and deployment with IAP authentication."
)(langgraph_demo.deploy)
