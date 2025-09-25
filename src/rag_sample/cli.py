#!/usr/bin/env python3
"""
Command line interface for RAG Sample application.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import readline
import atexit

from .rag_engine import RAGEngine
from .logging_config import get_logger
from .exceptions import RAGSampleError

logger = get_logger(__name__)
from .config import Config

console = Console()


class CommandHistory:
    """Manages command history for the CLI."""
    
    def __init__(self, max_history: int = 100):
        """Initialize command history.
        
        Args:
            max_history: Maximum number of commands to store
        """
        self.max_history = max_history
        self.history_file = Path.home() / ".rag_sample_history"
        self.history = []
        self.current_index = 0
        self._load_history()
        self._setup_readline()
    
    def _load_history(self):
        """Load command history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = [line.strip() for line in f.readlines() if line.strip()]
                # Keep only the most recent commands
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
        except Exception:
            self.history = []
    
    def _setup_readline(self):
        """Setup readline for command history."""
        try:
            readline.set_history_length(self.max_history)
            for command in self.history:
                readline.add_history(command)
        except Exception:
            pass
    
    def add_command(self, command: str):
        """Add command to history."""
        if command.strip() and command not in self.history:
            self.history.append(command)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            readline.add_history(command)
            self._save_history()
    
    def get_history(self):
        """Get command history."""
        return self.history.copy()
    
    def _save_history(self):
        """Save command history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                for command in self.history:
                    f.write(command + '\n')
        except Exception:
            pass


class CLIHandler:
    """Handles CLI operations and command processing."""
    
    def __init__(self, rag_engine, command_history, config_obj):
        """Initialize CLI handler.
        
        Args:
            rag_engine: RAG engine instance
            command_history: Command history instance
            config_obj: Configuration object
        """
        self.rag_engine = rag_engine
        self.command_history = command_history
        self.config_obj = config_obj
    
    def display_welcome_message(self):
        """Display welcome message and help information."""
        console.print(f"\n[yellow]{self.config_obj.welcome_message}[/yellow]")
        console.print("[dim]Type 'quit' or 'exit' to end the session[/dim]")
        console.print("[dim]Commands: /stats, /clear, /history, /add <file>, /docs, /docstats, /cmdhistory[/dim]")
        console.print("[dim]Document management: /remove <doc>, /listdocs, /testurl <url>, /reload[/dim]")
        console.print("[dim]Retrieval: /retrieval, /setretrieval <k> <threshold>, /debug[/dim]")
        console.print("[dim]UI: /setui <user_prompt> <goodbye_message>[/dim]")
        console.print("[dim]Web content: @<url> (e.g., @https://example.com)[/dim]")
        console.print("[dim]Navigation: ↑/↓ arrows for command history, Tab for completion[/dim]\n")
    
    def handle_stats_command(self):
        """Handle /stats command."""
        stats = self.rag_engine.get_conversation_stats()
        console.print(f"\n[bold cyan]Conversation Stats:[/bold cyan]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")
        console.print()
    
    def handle_clear_command(self):
        """Handle /clear command."""
        self.rag_engine.clear_conversation()
        console.print("\n[bold yellow]Conversation cleared![/bold yellow]\n")
    
    def handle_history_command(self):
        """Handle /history command."""
        messages = self.rag_engine.get_recent_messages(10)
        console.print(f"\n[bold cyan]Recent Messages:[/bold cyan]")
        for msg in messages:
            role_color = "green" if msg['role'] == 'user' else "blue"
            console.print(f"  [{role_color}]{msg['role'].title()}[/{role_color}]: {msg['content'][:100]}...")
        console.print()
    
    def handle_add_command(self, user_input):
        """Handle /add command."""
        file_path = user_input[5:].strip()
        if file_path:
            with console.status(f"[bold green]Adding document: {file_path}..."):
                success = self.rag_engine.add_document(file_path)
            if success:
                console.print(f"\n[bold green]✅ Document '{file_path}' added successfully![/bold green]\n")
            else:
                console.print(f"\n[bold red]❌ Failed to add document '{file_path}'[/bold red]\n")
        else:
            console.print(f"\n[bold yellow]Usage: /add <file_path>[/bold yellow]\n")
    
    def handle_url_command(self, user_input):
        """Handle @url command."""
        url = user_input[1:].strip()
        if url:
            with console.status(f"[bold green]Scraping and adding web content from: {url}..."):
                success = self.rag_engine.add_document_from_url(url)
            if success:
                console.print(f"\n[bold green]✅ Web content from '{url}' added successfully![/bold green]\n")
            else:
                console.print(f"\n[bold red]❌ Failed to add web content from '{url}'[/bold red]\n")
        else:
            console.print(f"\n[bold yellow]Usage: @<url>[/bold yellow]\n")
    
    def handle_docs_command(self):
        """Handle /docs command."""
        docs_path = self.rag_engine.config.documents_path
        console.print(f"\n[bold cyan]Documents Directory: {docs_path}[/bold cyan]")
        try:
            docs_dir = Path(docs_path)
            if docs_dir.exists():
                files = list(docs_dir.glob("*"))
                if files:
                    console.print(f"Found {len(files)} files:")
                    for i, file_path in enumerate(files, 1):
                        file_type = "📄" if file_path.is_file() else "📁"
                        console.print(f"  {i}. {file_type} {file_path.name}")
                else:
                    console.print("[yellow]No files found in documents directory[/yellow]")
            else:
                console.print(f"[yellow]Documents directory does not exist: {docs_path}[/yellow]")
        except Exception as e:
            console.print(f"[red]Error reading documents directory: {str(e)}[/red]")
        console.print()
    
    def handle_docstats_command(self):
        """Handle /docstats command."""
        stats = self.rag_engine.get_document_stats()
        console.print(f"\n[bold cyan]Document Stats:[/bold cyan]")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")
        console.print()
    
    def handle_cmdhistory_command(self):
        """Handle /cmdhistory command."""
        history = self.command_history.get_history()
        if history:
            console.print(f"\n[bold cyan]Command History:[/bold cyan]")
            for i, cmd in enumerate(history[-10:], 1):  # Show last 10 commands
                console.print(f"  {i}. {cmd}")
        else:
            console.print("[yellow]No command history[/yellow]")
        console.print()
    
    def handle_remove_command(self, user_input):
        """Handle /remove command."""
        doc_name = user_input[8:].strip()
        if doc_name:
            try:
                result = self.rag_engine.remove_document(doc_name)
                console.print(f"[green]Document removed: {result}[/green]")
            except Exception as e:
                console.print(f"[red]Error removing document: {str(e)}[/red]")
        else:
            console.print("[red]Please specify a document name[/red]")
    
    def handle_testurl_command(self, user_input):
        """Handle /testurl command."""
        url = user_input[9:].strip()
        if url:
            try:
                result = self.rag_engine.test_url(url)
                console.print(f"[green]URL test successful: {result}[/green]")
            except Exception as e:
                console.print(f"[red]Error testing URL: {str(e)}[/red]")
        else:
            console.print("[red]Please specify a URL[/red]")
    
    def handle_reload_command(self):
        """Handle /reload command."""
        try:
            result = self.rag_engine.reload_documents()
            console.print(f"[green]Documents reloaded: {result}[/green]")
        except Exception as e:
            console.print(f"[red]Error reloading documents: {str(e)}[/red]")
    
    def handle_retrieval_command(self):
        """Handle /retrieval command."""
        settings = self.rag_engine.get_retrieval_settings()
        console.print(f"\n[bold cyan]Retrieval Settings:[/bold cyan]")
        console.print(f"  Top K: {settings.get('top_k', 'N/A')}")
        console.print(f"  Threshold: {settings.get('threshold', 'N/A')}")
        console.print(f"  Debug: {settings.get('debug', 'N/A')}")
        console.print()
    
    def handle_setretrieval_command(self, user_input):
        """Handle /setretrieval command."""
        parts = user_input[14:].strip().split()
        if len(parts) == 2:
            try:
                k = int(parts[0])
                threshold = float(parts[1])
                self.rag_engine.set_retrieval_settings(k, threshold)
                console.print(f"[green]Retrieval settings updated: k={k}, threshold={threshold}[/green]")
            except ValueError:
                console.print("[red]Invalid values. Use: /setretrieval <k> <threshold>[/red]")
        else:
            console.print("[red]Usage: /setretrieval <k> <threshold>[/red]")
    
    def handle_debug_command(self):
        """Handle /debug command."""
        self.rag_engine.toggle_debug()
        console.print(f"[green]Debug mode: {'ON' if self.rag_engine.debug_enabled else 'OFF'}[/green]")
    
    def handle_setui_command(self, user_input):
        """Handle /setui command."""
        parts = user_input[7:].strip().split(' ', 1)
        if len(parts) == 2:
            user_prompt, goodbye_msg = parts
            self.rag_engine.set_ui_settings(user_prompt, goodbye_msg)
            console.print(f"[green]UI settings updated[/green]")
        else:
            console.print("[red]Usage: /setui <user_prompt> <goodbye_message>[/red]")
    
    def process_user_input(self, user_input):
        """Process user input and handle commands."""
        # Add command to history (except for special commands)
        if user_input.strip() and not user_input.lower() in ['quit', 'exit', 'q']:
            self.command_history.add_command(user_input)
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print(f"\n[yellow]{self.config_obj.goodbye_message}[/yellow]")
            return False
        
        # Handle special commands
        if user_input.lower() in ['/stats', '/status']:
            self.handle_stats_command()
            return True
        
        if user_input.lower() in ['/clear', '/reset']:
            self.handle_clear_command()
            return True
        
        if user_input.lower() in ['/history', '/messages']:
            self.handle_history_command()
            return True
        
        if user_input.lower().startswith('/add '):
            self.handle_add_command(user_input)
            return True
        
        # Handle @url syntax for adding web documents
        if user_input.startswith('@'):
            self.handle_url_command(user_input)
            return True
        
        if user_input.lower() in ['/docs', '/documents']:
            self.handle_docs_command()
            return True
        
        if user_input.lower() in ['/docstats', '/documents-stats']:
            self.handle_docstats_command()
            return True
        
        if user_input.lower() in ['/cmdhistory', '/history-cmd']:
            self.handle_cmdhistory_command()
            return True
        
        if user_input.lower().startswith('/remove '):
            self.handle_remove_command(user_input)
            return True
        
        if user_input.lower() in ['/listdocs', '/list-documents']:
            self.handle_docs_command()
            return True
        
        if user_input.lower().startswith('/testurl '):
            self.handle_testurl_command(user_input)
            return True
        
        if user_input.lower() in ['/reload', '/reload-docs']:
            self.handle_reload_command()
            return True
        
        if user_input.lower() in ['/retrieval', '/retrieval-settings']:
            self.handle_retrieval_command()
            return True
        
        if user_input.lower().startswith('/setretrieval '):
            self.handle_setretrieval_command(user_input)
            return True
        
        if user_input.lower() in ['/debug', '/debug-mode']:
            self.handle_debug_command()
            return True
        
        if user_input.lower().startswith('/setui '):
            self.handle_setui_command(user_input)
            return True
        
        # Regular chat
        if user_input.strip():
            try:
                response = self.rag_engine.chat(user_input)
                console.print(f"\n[bold blue]🤖 Response:[/bold blue] {response}\n")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
        
        return True


@click.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--model', '-m', default='llama3-8b-8192', help='Groq model to use (llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768)')
@click.option('--temperature', '-t', default=0.7, help='Temperature for response generation')
@click.option('--max-tokens', default=1000, help='Maximum tokens for response')
def main(config, model, temperature, max_tokens):
    """RAG Sample - Chat with your documents using RAG."""
    
    console.print(Panel.fit(
        "[bold blue]RAG Sample[/bold blue]\n"
        "A conversational RAG application for document chat",
        title="Welcome"
    ))
    
    try:
        config_obj = Config(config_path=config)
        rag_engine = RAGEngine(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            config=config_obj
        )
        
        # Initialize command history
        command_history = CommandHistory()
        
        # Initialize CLI handler
        cli_handler = CLIHandler(rag_engine, command_history, config_obj)
        
        cli_handler.display_welcome_message()
        
        # Interactive chat loop
        while True:
            try:
                user_input = Prompt.ask(f"[bold green]{config_obj.user_prompt}[/bold green]")
                
                if not cli_handler.process_user_input(user_input):
                    break
                
            except KeyboardInterrupt:
                console.print(f"\n[yellow]{config_obj.goodbye_message}[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {str(e)}[/red]")
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
    
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG engine: {str(e)}[/red]")
        logger.error(f"Failed to initialize RAG engine: {e}", exc_info=True)
        return 1
    
    return 0
