#!/usr/bin/env python3
"""
Command line interface for RAG Sample application.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from .rag_engine import RAGEngine
from .config import Config

console = Console()


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
        # Initialize configuration
        config_obj = Config(config_path=config)
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            config=config_obj
        )
        
        console.print("\n[yellow]Starting interactive chat session...[/yellow]")
        console.print("[dim]Type 'quit' or 'exit' to end the session[/dim]")
        console.print("[dim]Commands: /stats, /clear, /history, /add <file>, /docs, /docstats[/dim]")
        console.print("[dim]Web content: @<url> (e.g., @https://example.com)[/dim]\n")
        
        # Interactive chat loop
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[yellow]Goodbye![/yellow]")
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
                    console.print()
                    continue
                
                if user_input.lower() in ['/docstats', '/docstats']:
                    stats = rag_engine.get_document_stats()
                    console.print(f"\n[bold cyan]Document Statistics:[/bold cyan]")
                    for key, value in stats.items():
                        console.print(f"  {key}: {value}")
                    console.print()
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
