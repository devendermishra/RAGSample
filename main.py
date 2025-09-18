#!/usr/bin/env python3
"""
RAG Sample CLI Application

A command-line tool for document preparation and conversational RAG.
"""

import click
import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import the full versions, fall back to demo versions
try:
    from ragsample.prepare import prepare_documents
    FULL_VERSION = True
except ImportError:
    from ragsample.prepare_demo import prepare_documents
    FULL_VERSION = False

try:
    from ragsample.chat import start_chat
except ImportError:
    from ragsample.chat_demo import start_chat


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RAG Sample - A CLI-based conversational RAG app.
    
    This application allows you to:
    1. Prepare documents by downloading and storing them in a vector database
    2. Chat with the stored documents using conversational AI
    """
    if not FULL_VERSION:
        click.echo("⚠️  Running in demo mode. For full functionality, install:")
        click.echo("   pip install langchain langchain-openai langchain-community chromadb beautifulsoup4")
        click.echo()


@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--db-path', default='./rag_db', help='Path to store the vector database')
def prepare(urls, db_path):
    """Download and prepare documents for RAG.
    
    URLS: One or more URLs to download and process
    """
    click.echo(f"Preparing documents from {len(urls)} URL(s)...")
    click.echo(f"Database path: {db_path}")
    
    try:
        prepare_documents(urls, db_path)
        click.echo("✅ Documents prepared successfully!")
    except Exception as e:
        click.echo(f"❌ Error preparing documents: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--db-path', default='./rag_db', help='Path to the vector database')
def chat(db_path):
    """Start an interactive chat session with your documents."""
    if FULL_VERSION:
        # Check for vector database
        if not os.path.exists(db_path):
            click.echo(f"❌ Database not found at {db_path}. Please run 'prepare' first.", err=True)
            sys.exit(1)
    else:
        # Check for demo database
        documents_file = Path(db_path) / "documents.json"
        if not documents_file.exists():
            click.echo(f"❌ Documents not found at {db_path}. Please run 'prepare' first.", err=True)
            sys.exit(1)
    
    click.echo("Starting chat session...")
    click.echo("Type 'quit' or 'exit' to end the session.")
    click.echo("-" * 50)
    
    try:
        start_chat(db_path)
    except Exception as e:
        click.echo(f"❌ Error during chat: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()