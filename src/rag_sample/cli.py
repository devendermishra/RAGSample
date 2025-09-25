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
    
    def _save_history(self):
        """Save command history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                for command in self.history:
                    f.write(f"{command}\n")
        except Exception:
            pass  # Silently fail if we can't save history
    
    def _setup_readline(self):
        """Setup readline for command history."""
        try:
            # Set up readline history
            readline.clear_history()
            for command in self.history:
                readline.add_history(command)
            
            # Set up tab completion for common commands
            readline.set_completer(self._completer)
            readline.parse_and_bind("tab: complete")
            
            # Register cleanup function
            atexit.register(self._save_history)
        except Exception:
            pass  # Silently fail if readline is not available
    
    def _completer(self, text, state):
        """Tab completion for commands."""
        commands = [
            '/stats', '/status', '/clear', '/reset', '/history', '/messages',
            '/add', '/docs', '/documents', '/docstats', '/help', '/quit', '/exit',
            '/remove', '/listdocs', '/list', '/cmdhistory', '/cmds', '/clearhistory', '/clearhist',
            '/retrieval', '/retrievalconfig', '/setretrieval', '/debug', '/toggledebug',
            '/ui', '/uiconfig', '/setui', '/reload', '/reloaddocs'
        ]
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None
    
    def add_command(self, command: str):
        """Add a command to history.
        
        Args:
            command: Command to add
        """
        if command.strip() and (not self.history or self.history[-1] != command):
            self.history.append(command)
            # Keep only the most recent commands
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            # Add to readline history
            try:
                readline.add_history(command)
            except Exception:
                pass
    
    def get_history(self, limit: int = 10) -> list:
        """Get recent command history.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of recent commands
        """
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear command history."""
        self.history.clear()
        try:
            readline.clear_history()
        except Exception:
            pass
        # Remove history file
        try:
            if self.history_file.exists():
                self.history_file.unlink()
        except Exception:
            pass


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
        
        console.print(f"\n[yellow]{config_obj.welcome_message}[/yellow]")
        console.print("[dim]Type 'quit' or 'exit' to end the session[/dim]")
        console.print("[dim]Commands: /stats, /clear, /history, /add <file>, /docs, /docstats, /cmdhistory[/dim]")
        console.print("[dim]Document management: /remove <doc>, /listdocs, /testurl <url>, /reload[/dim]")
        console.print("[dim]Retrieval: /retrieval, /setretrieval <k> <threshold>, /debug[/dim]")
        console.print("[dim]UI: /setui <user_prompt> <goodbye_message>[/dim]")
        console.print("[dim]Web content: @<url> (e.g., @https://example.com)[/dim]")
        console.print("[dim]Navigation: ↑/↓ arrows for command history, Tab for completion[/dim]\n")
        
        # Interactive chat loop
        while True:
            try:
                user_input = Prompt.ask(f"[bold green]{config_obj.user_prompt}[/bold green]")
                
                # Add command to history (except for special commands)
                if user_input.strip() and not user_input.lower() in ['quit', 'exit', 'q']:
                    command_history.add_command(user_input)
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print(f"\n[yellow]{config_obj.goodbye_message}[/yellow]")
                    break
                
                # Handle special commands
                if user_input.lower() in ['/stats', '/status']:
                    stats = rag_engine.get_conversation_stats()
                    console.print(f"\n[bold cyan]Conversation Stats:[/bold cyan]")
                    for key, value in stats.items():
                        console.print(f"  {key}: {value}")
                    console.print()
                    continue
                
                if user_input.lower() in ['/clear', '/reset']:
                    rag_engine.clear_conversation()
                    console.print("\n[bold yellow]Conversation cleared![/bold yellow]\n")
                    continue
                
                if user_input.lower() in ['/history', '/messages']:
                    messages = rag_engine.get_recent_messages(10)
                    console.print(f"\n[bold cyan]Recent Messages:[/bold cyan]")
                    for msg in messages:
                        role_color = "green" if msg['role'] == 'user' else "blue"
                        console.print(f"  [{role_color}]{msg['role'].title()}[/{role_color}]: {msg['content'][:100]}...")
                    console.print()
                    continue
                
                if user_input.lower().startswith('/add '):
                    file_path = user_input[5:].strip()
                    if file_path:
                        with console.status(f"[bold green]Adding document: {file_path}..."):
                            success = rag_engine.add_document(file_path)
                        if success:
                            console.print(f"\n[bold green]✅ Document '{file_path}' added successfully![/bold green]\n")
                        else:
                            console.print(f"\n[bold red]❌ Failed to add document '{file_path}'[/bold red]\n")
                    else:
                        console.print(f"\n[bold yellow]Usage: /add <file_path>[/bold yellow]\n")
                    continue
                
                # Handle @url syntax for adding web documents
                if user_input.startswith('@'):
                    url = user_input[1:].strip()
                    if url:
                        with console.status(f"[bold green]Scraping and adding web content from: {url}..."):
                            success = rag_engine.add_document_from_url(url)
                        if success:
                            console.print(f"\n[bold green]✅ Web content from '{url}' added successfully![/bold green]\n")
                        else:
                            console.print(f"\n[bold red]❌ Failed to add web content from '{url}'[/bold red]\n")
                    else:
                        console.print(f"\n[bold yellow]Usage: @<url>[/bold yellow]\n")
                    continue
                
                if user_input.lower() in ['/docs', '/documents']:
                    docs_path = rag_engine.config.documents_path
                    console.print(f"\n[bold cyan]Documents Directory: {docs_path}[/bold cyan]")
                    try:
                        docs_dir = Path(docs_path)
                        if docs_dir.exists():
                            files = list(docs_dir.rglob("*"))
                            supported_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md']]
                            if supported_files:
                                console.print(f"[green]Found {len(supported_files)} supported documents:[/green]")
                                for file in supported_files:
                                    size = file.stat().st_size
                                    console.print(f"  📄 {file.name} ({size:,} bytes)")
                            else:
                                console.print("[yellow]No supported documents found.[/yellow]")
                        else:
                            console.print(f"[yellow]Documents directory does not exist: {docs_path}[/yellow]")
                    except Exception as e:
                        console.print(f"[red]Error reading documents directory: {e}[/red]")
                    
                    # Also show documents from vector store
                    console.print(f"\n[bold cyan]Documents in Vector Store:[/bold cyan]")
                    try:
                        documents = rag_engine.list_documents()
                        if documents:
                            console.print(f"[green]Found {len(documents)} documents in vector store:[/green]")
                            for i, doc in enumerate(documents, 1):
                                source = doc.get('source', 'Unknown')
                                title = doc.get('title', 'No title')
                                doc_type = doc.get('type', 'unknown')
                                chunk_count = doc.get('chunk_count', 0)
                                domain = doc.get('domain', '')
                                
                                # Determine icon based on type
                                if doc_type == 'web_page':
                                    icon = "🌐"
                                elif doc_type == 'pdf':
                                    icon = "📄"
                                elif doc_type == 'txt':
                                    icon = "📝"
                                else:
                                    icon = "📄"
                                
                                console.print(f"  {i:2d}. {icon} [bold]{title}[/bold]")
                                console.print(f"      Source: {source}")
                                console.print(f"      Type: {doc_type}")
                                console.print(f"      Chunks: {chunk_count}")
                                if domain:
                                    console.print(f"      Domain: {domain}")
                                console.print()
                        else:
                            console.print("[yellow]No documents found in vector store.[/yellow]")
                    except Exception as e:
                        console.print(f"[red]Error reading vector store: {e}[/red]")
                    
                    console.print()
                    continue
                
                if user_input.lower() in ['/docstats', '/docstats']:
                    stats = rag_engine.get_document_stats()
                    console.print(f"\n[bold cyan]Document Statistics:[/bold cyan]")
                    for key, value in stats.items():
                        console.print(f"  {key}: {value}")
                    console.print()
                    continue
                
                if user_input.lower() in ['/cmdhistory', '/cmds']:
                    history = command_history.get_history(20)
                    console.print(f"\n[bold cyan]Command History (last 20):[/bold cyan]")
                    if history:
                        for i, cmd in enumerate(history, 1):
                            console.print(f"  {i:2d}. {cmd}")
                    else:
                        console.print("  No command history found.")
                    console.print()
                    continue
                
                if user_input.lower() in ['/clearhistory', '/clearhist']:
                    command_history.clear_history()
                    console.print("\n[bold yellow]Command history cleared![/bold yellow]\n")
                    continue
                
                if user_input.lower().startswith('/remove '):
                    document_identifier = user_input[8:].strip()
                    if document_identifier:
                        with console.status(f"[bold red]Removing document: {document_identifier}..."):
                            success = rag_engine.remove_document(document_identifier)
                        if success:
                            console.print(f"\n[bold green]✅ Document '{document_identifier}' removed successfully![/bold green]\n")
                        else:
                            console.print(f"\n[bold red]❌ Failed to remove document '{document_identifier}'[/bold red]\n")
                    else:
                        console.print(f"\n[bold yellow]Usage: /remove <document_identifier>[/bold yellow]\n")
                    continue
                
                if user_input.lower() in ['/listdocs', '/list']:
                    documents = rag_engine.list_documents()
                    console.print(f"\n[bold cyan]Documents in Vector Store:[/bold cyan]")
                    if documents:
                        for i, doc in enumerate(documents, 1):
                            source = doc.get('source', 'Unknown')
                            title = doc.get('title', 'No title')
                            doc_type = doc.get('type', 'unknown')
                            chunk_count = doc.get('chunk_count', 0)
                            domain = doc.get('domain', '')
                            
                            console.print(f"  {i:2d}. [bold]{title}[/bold]")
                            console.print(f"      Source: {source}")
                            console.print(f"      Type: {doc_type}")
                            console.print(f"      Chunks: {chunk_count}")
                            if domain:
                                console.print(f"      Domain: {domain}")
                            console.print()
                    else:
                        console.print("  No documents found in the vector store.")
                    console.print()
                    continue
                
                if user_input.lower().startswith('/testurl '):
                    url = user_input[9:].strip()
                    if url:
                        console.print(f"\n[bold cyan]Testing web scraping for: {url}[/bold cyan]")
                        result = rag_engine.web_scraper.test_scraping(url)
                        if result["success"]:
                            console.print(f"[green]✅ Successfully scraped {result['content_length']} characters[/green]")
                            console.print(f"[green]Title: {result['title']}[/green]")
                            console.print(f"[green]Domain: {result['domain']}[/green]")
                            console.print(f"[green]Preview: {result['preview'][:200]}...[/green]")
                        else:
                            console.print(f"[red]❌ Failed: {result['error']}[/red]")
                        console.print()
                    else:
                        console.print(f"\n[bold yellow]Usage: /testurl <url>[/bold yellow]\n")
                    continue
                
                if user_input.lower() in ['/retrieval', '/retrievalconfig']:
                    console.print(f"\n[bold cyan]Retrieval Configuration:[/bold cyan]")
                    console.print(f"  Top K: {rag_engine.config.retrieval_top_k}")
                    console.print(f"  Threshold: {rag_engine.config.retrieval_threshold}")
                    console.print(f"  Debug Mode: {rag_engine.config.enable_retrieval_debug}")
                    console.print()
                    continue
                
                if user_input.lower().startswith('/setretrieval '):
                    params = user_input[14:].strip().split()
                    if len(params) >= 2:
                        try:
                            top_k = int(params[0])
                            threshold = float(params[1])
                            rag_engine.config.retrieval_top_k = top_k
                            rag_engine.config.retrieval_threshold = threshold
                            console.print(f"\n[bold green]✅ Retrieval settings updated:[/bold green]")
                            console.print(f"  Top K: {top_k}")
                            console.print(f"  Threshold: {threshold}")
                            console.print()
                        except ValueError:
                            console.print(f"\n[bold red]❌ Invalid parameters. Usage: /setretrieval <top_k> <threshold>[/bold red]\n")
                    else:
                        console.print(f"\n[bold yellow]Usage: /setretrieval <top_k> <threshold>[/bold yellow]\n")
                    continue
                
                if user_input.lower() in ['/toggledebug', '/debug']:
                    rag_engine.config.enable_retrieval_debug = not rag_engine.config.enable_retrieval_debug
                    status = "enabled" if rag_engine.config.enable_retrieval_debug else "disabled"
                    console.print(f"\n[bold green]✅ Retrieval debug mode {status}[/bold green]\n")
                    continue
                
                
                if user_input.lower() in ['/ui', '/uiconfig']:
                    console.print(f"\n[bold cyan]UI Configuration:[/bold cyan]")
                    console.print(f"  User Prompt: '{config_obj.user_prompt}'")
                    console.print(f"  Goodbye Message: '{config_obj.goodbye_message}'")
                    console.print(f"  Welcome Message: '{config_obj.welcome_message}'")
                    console.print()
                    continue
                
                if user_input.lower() in ['/reload', '/reloaddocs']:
                    with console.status("[bold green]Reloading documents..."):
                        success = rag_engine.reload_documents()
                    if success:
                        console.print(f"\n[bold green]✅ Documents reloaded successfully![/bold green]\n")
                    else:
                        console.print(f"\n[bold red]❌ Failed to reload documents[/bold red]\n")
                    continue
                
                if user_input.lower().startswith('/setui '):
                    params = user_input[7:].strip().split('|', 1)  # Use | as separator for multi-word messages
                    if len(params) >= 2:
                        try:
                            user_prompt = params[0].strip()
                            goodbye_message = params[1].strip()
                            config_obj.user_prompt = user_prompt
                            config_obj.goodbye_message = goodbye_message
                            console.print(f"\n[bold green]✅ UI settings updated:[/bold green]")
                            console.print(f"  User Prompt: '{user_prompt}'")
                            console.print(f"  Goodbye Message: '{goodbye_message}'")
                            console.print()
                        except Exception as e:
                            console.print(f"\n[bold red]❌ Error updating UI settings: {str(e)}[/bold red]\n")
                    else:
                        console.print(f"\n[bold yellow]Usage: /setui <user_prompt>|<goodbye_message>[/bold yellow]")
                        console.print(f"[dim]Example: /setui 'User'|'See you later!'[/dim]\n")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Get response from RAG engine
                with console.status("[bold green]Thinking..."):
                    response = rag_engine.chat(user_input)
                
                # Display response
                console.print(f"\n[bold blue]Assistant[/bold blue]: {response}\n")
                
                # Show conversation stats if memory is enabled
                if rag_engine.conversation_memory:
                    stats = rag_engine.get_conversation_stats()
                    if stats.get('token_usage_percentage', 0) > 70:
                        console.print(f"[dim]💡 Conversation memory: {stats.get('token_usage_percentage', 0):.1f}% used[/dim]\n")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                
    except Exception as e:
        console.print(f"[red]Failed to initialize RAG engine: {str(e)}[/red]")
        console.print("[yellow]Please check your configuration and try again.[/yellow]")


if __name__ == "__main__":
    main()
