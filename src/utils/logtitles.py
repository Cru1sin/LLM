from rich.console import Console

# console = Console()

def log_title(message: str):
    console = Console()
    total_length = 80
    message_length = len(message)
    padding = max(0, total_length - message_length - 2)
    left_padding = "=" * (padding // 2)
    right_padding = "=" * (padding - len(left_padding))
    padded_message = f"{left_padding} {message} {right_padding}"
    
    console.print(padded_message, style="bold cyan")

if __name__ == "__main__":
    log_title("Hello, World!")