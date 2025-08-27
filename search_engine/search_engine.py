import joblib
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk  # Added this import to fix the NameError
from tkinter import scrolledtext
from tkinter import font as tkfont
from sklearn.metrics.pairwise import cosine_similarity
import threading
import webbrowser

INDEX_FILE = './data/index.pkl'

class Ranker:
    def __init__(self, vectorizer, tfidf_matrix):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

    def rank(self, query: str) -> list[tuple[int, float]]:
        if not query:
            return []

        query_vector = self.vectorizer.transform([query])
        
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        doc_scores = [
            (doc_id, score) for doc_id, score in enumerate(cosine_scores) if score > 0
        ]
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores

class VerticalSearchEngine:
    def __init__(self):
        print("Loading index...")
        try:
            index_data = joblib.load(INDEX_FILE)
            self.positional_index = index_data['positional_index']
            self.doc_store = index_data['doc_store']
            self.ranker = Ranker(index_data['vectorizer'], index_data['tfidf_matrix'])
            print("Index loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Index file not found at {INDEX_FILE}.")
            print("Please run the indexer first using 'python main.py index'.")
            exit()
        
    def search(self, query: str, top_k: int = 1000):
        scores = []
        search_query = query

        scores = self.ranker.rank(search_query)
        
        results = []
        for doc_id, score in scores[:top_k]:
            doc_info = self.doc_store[doc_id]
            
            formatted_authors = [
                {"name": author}
                for author in doc_info.get("authors", [])
            ]
            
            publication = {
                "title": doc_info.get("title", "No Title"),
                "authors": formatted_authors,
                "abstract": doc_info.get("abstract", ""),
                "published_date": doc_info.get("published_date", "N/A"),
                "link": doc_info.get("link", ""),
                "relevancy_score": round(score, 4)
            }
            results.append(publication)
        return results

class SearchUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tb.Window(themename="cosmo")  # Use ttkbootstrap Window
        self.root.title("Coventry University Research Publications")
        self.root.geometry("1400x900")
        self.links = {}
        
        # Configure custom fonts
        self.title_font = tkfont.Font(family="Segoe UI", size=28, weight="bold")
        self.subtitle_font = tkfont.Font(family="Segoe UI", size=15)
        self.text_font = tkfont.Font(family="Segoe UI", size=12)
        self.result_title_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        
        self.setup_theme()
        self.create_widgets()
        
    def setup_theme(self):
        # Define colors using ttkbootstrap's style system
        style = self.root.style
        self.bg_color = style.colors.light
        self.fg_color = style.colors.dark
        self.entry_bg = style.colors.light
        self.result_bg = style.colors.light
        self.highlight_bg = style.colors.secondary
        self.accent_color = style.colors.primary
        self.accent_hover = style.colors.info
        self.border_color = style.colors.secondary
        
        self.root.configure(bg=self.bg_color)
        
    def create_widgets(self):
        # Main container
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Header
        header = tk.Frame(main_container, bg=self.bg_color)
        header.pack(fill=tk.X, pady=(0, 30))
        
        title = tk.Label(
            header,
            text="Research Publications Search",
            font=self.title_font,
            bg=self.bg_color,
            fg=self.accent_color
        )
        title.pack()
        
        subtitle = tk.Label(
            header,
            text="Search through Coventry University's research publications",
            font=self.subtitle_font,
            bg=self.bg_color,
            fg=self.fg_color
        )
        subtitle.pack(pady=(5, 0))
        
        # Search container
        search_container = tk.Frame(main_container, bg=self.bg_color)
        search_container.pack(fill=tk.X, pady=(0, 20))
        
        # Search entry with border
        entry_frame = tk.Frame(search_container, bg=self.accent_color, padx=2, pady=2)
        entry_frame.pack(fill=tk.X)
        
        self.search_entry = tk.Entry(
            entry_frame,
            font=self.text_font,
            bg=self.entry_bg,
            fg=self.fg_color,
            insertbackground=self.fg_color,
            relief=tk.FLAT
        )
        self.search_entry.pack(fill=tk.X, ipady=8)
        self.search_entry.bind('<Return>', self.perform_search)
        
        # Buttons container
        buttons_frame = tk.Frame(search_container, bg=self.bg_color)
        buttons_frame.pack(pady=15)
        
        # Search button with hover effect
        self.search_button = tk.Button(
            buttons_frame,
            text="Search",
            font=self.text_font,
            bg=self.accent_color,
            fg="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2"
        )
        self.search_button.pack(side=tk.LEFT, padx=5)
        self.search_button.bind('<Button-1>', self.perform_search)
        self.search_button.bind('<Enter>', lambda e: e.widget.configure(bg=self.darken_color(self.accent_color)))
        self.search_button.bind('<Leave>', lambda e: e.widget.configure(bg=self.accent_color))
        
        # Clear button
        self.clear_button = tk.Button(
            buttons_frame,
            text="Clear",
            font=self.text_font,
            bg=self.highlight_bg,
            fg=self.fg_color,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.clear_search
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Loading indicator
        self.loading_label = tk.Label(
            search_container,
            text="Searching...",
            font=self.text_font,
            bg=self.bg_color,
            fg=self.accent_color
        )
        
        # Results area with custom styling
        self.results_text = scrolledtext.ScrolledText(
            main_container,
            font=self.text_font,
            bg=self.result_bg,
            fg=self.fg_color,
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            cursor="arrow"  # Default cursor
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.tag_configure("title", font=self.result_title_font, foreground=self.accent_color)
        self.results_text.tag_configure("heading", font=self.subtitle_font, foreground=self.fg_color)
        self.results_text.tag_configure("link", foreground="blue", underline=True)
        
        # Bind click and hover events for links
        self.results_text.tag_bind("link", "<Button-1>", self.open_url)
        self.results_text.tag_bind("link", "<Enter>", lambda e: self.results_text.configure(cursor="hand2"))
        self.results_text.tag_bind("link", "<Leave>", lambda e: self.results_text.configure(cursor="arrow"))

    def darken_color(self, color):
        # Darken a hex color by 20%
        r = int(color[1:3], 16) * 0.8
        g = int(color[3:5], 16) * 0.8
        b = int(color[5:7], 16) * 0.8
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        
    def clear_search(self):
        self.search_entry.delete(0, tk.END)
        self.results_text.delete(1.0, tk.END)
        
    def perform_search(self, event=None):
        query = self.search_entry.get().strip()
        if not query:
            return
            
        self.loading_label.pack(pady=5)
        self.results_text.delete(1.0, tk.END)
        self.root.update()
        
        def search():
            results = self.engine.search(query)
            self.root.after(0, lambda: self.display_results(results, query))
        
        threading.Thread(target=search, daemon=True).start()
        
    def open_url(self, event):
        for start, end, url in self.links.values():
            if self.results_text.compare(f"@{event.x},{event.y}", ">=", start) and \
               self.results_text.compare(f"@{event.x},{event.y}", "<=", end):
                webbrowser.open_new_tab(url)
                break

    def display_results(self, results, query):
        self.loading_label.pack_forget()
        self.links.clear()  # Clear previous links
        
        if not results:
            self.results_text.insert(tk.END, "No results found.", "heading")
            return
            
        self.results_text.insert(tk.END, f"Found {len(results)} results for '{query}'\n\n", "heading")
        
        for i, res in enumerate(results, 1):
            self.results_text.insert(tk.END, f"{i}. ", "heading")
            self.results_text.insert(tk.END, f"{res['title']}\n", "title")
            self.results_text.insert(tk.END, f"Published: {res['published_date']}\n\n")
            self.results_text.insert(tk.END, "Authors:\n", "heading")
            for author in res['authors']:
                self.results_text.insert(tk.END, f"  • {author['name']}\n")
            self.results_text.insert(tk.END, "\nAbstract:\n", "heading")
            self.results_text.insert(tk.END, f"{res['abstract']}\n\n")
            
            # Insert URL as a clickable link
            self.results_text.insert(tk.END, "URL: ")
            url_start = self.results_text.index("end-1c")
            self.results_text.insert(tk.END, f"{res['link']}", "link")
            url_end = self.results_text.index("end-1c")
            self.links[i] = (url_start, url_end, res['link'])
            self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, f"Relevancy Score: {res['relevancy_score']}\n")
            self.results_text.insert(tk.END, "─" * 80 + "\n\n")
            
    def run(self):
        # Center the window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the main event loop
        self.root.mainloop()

if __name__ == '__main__':
    engine = VerticalSearchEngine()
    app = SearchUI(engine)
    app.run()
    