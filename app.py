import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import re
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import hashlib
import time
import requests
import urllib.parse
from xml.sax.saxutils import unescape
import html

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class YouTubeSummarizer:
    def __init__(self):
        self.model = genai.GenerativeModel('models/learnlm-2.0-flash-experimental')
        self.cache_dir = "transcript_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_video_metadata(self, video_url):
        """Extract video metadata using yt-dlp"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                metadata = {
                    'title': info.get('title', 'Unknown Title'),
                    'channel': info.get('uploader', 'Unknown Channel'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', '')[:500] + '...' if info.get('description') else ''
                }
                
                # Format duration
                if metadata['duration']:
                    duration = str(timedelta(seconds=metadata['duration']))
                    metadata['formatted_duration'] = duration
                else:
                    metadata['formatted_duration'] = 'Unknown'
                
                # Format upload date
                if metadata['upload_date']:
                    try:
                        date_obj = datetime.strptime(metadata['upload_date'], '%Y%m%d')
                        metadata['formatted_upload_date'] = date_obj.strftime('%B %d, %Y')
                    except:
                        metadata['formatted_upload_date'] = metadata['upload_date']
                else:
                    metadata['formatted_upload_date'] = 'Unknown'
                
                return metadata
                
        except Exception as e:
            st.error(f"Error extracting metadata: {str(e)}")
            return None
    
    def get_transcript_ytdlp(self, video_url):
        """Get transcript using yt-dlp as fallback method"""
        try:
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Check for subtitles
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Try different subtitle languages
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in subtitles:
                        # Manual subtitles preferred
                        subtitle_entries = subtitles[lang]
                        for entry in subtitle_entries:
                            if 'url' in entry:
                                content = self._download_subtitle_content(entry['url'])
                                if content:
                                    return content, None
                    
                    if lang in automatic_captions:
                        # Auto-generated captions as fallback
                        caption_entries = automatic_captions[lang]
                        for entry in caption_entries:
                            if 'url' in entry:
                                content = self._download_subtitle_content(entry['url'])
                                if content:
                                    return content, None
                
                return None, "No English subtitles/captions found"
                
        except Exception as e:
            return None, f"yt-dlp transcript error: {str(e)}"
    
    def _download_subtitle_content(self, subtitle_url):
        """Download and parse subtitle content with multiple methods"""
        try:
            # Method 1: Direct requests
            response = requests.get(subtitle_url, timeout=30)
            response.raise_for_status()
            content = response.text
            
            # Clean and parse content
            transcript_text = self._parse_subtitle_content(content)
            return transcript_text
            
        except Exception as e:
            try:
                # Method 2: urllib as fallback
                import urllib.request
                response = urllib.request.urlopen(subtitle_url)
                content = response.read().decode('utf-8')
                transcript_text = self._parse_subtitle_content(content)
                return transcript_text
            except Exception as e2:
                raise Exception(f"Error downloading subtitles: {str(e)} | Fallback error: {str(e2)}")
    
    def _parse_subtitle_content(self, content):
        """Parse subtitle content using multiple approaches"""
        transcript_text = ""
        
        try:
            # Method 1: Try XML parsing
            import xml.etree.ElementTree as ET
            
            # Clean the content first
            content = content.strip()
            if not content:
                raise ValueError("Empty content")
            
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Parse XML
            root = ET.fromstring(content)
            
            # Extract text from different XML structures
            for text_elem in root.findall('.//text'):
                if text_elem.text:
                    clean_text = html.unescape(text_elem.text)
                    transcript_text += clean_text + " "
            
            # If no text found, try different tags
            if not transcript_text.strip():
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        clean_text = html.unescape(elem.text)
                        transcript_text += clean_text + " "
            
            if transcript_text.strip():
                return transcript_text.strip()
            else:
                raise ValueError("No text content found in XML")
                
        except Exception as xml_error:
            # Method 2: Regex-based extraction as fallback
            try:
                # Remove HTML/XML tags and extract text
                text_pattern = r'<[^>]*?>(.*?)</[^>]*?>'
                matches = re.findall(text_pattern, content, re.DOTALL)
                
                if matches:
                    for match in matches:
                        clean_text = html.unescape(match.strip())
                        if clean_text:
                            transcript_text += clean_text + " "
                
                # If still no content, try simpler extraction
                if not transcript_text.strip():
                    # Remove all HTML/XML tags and get remaining text
                    clean_content = re.sub(r'<[^>]+>', ' ', content)
                    clean_content = html.unescape(clean_content)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    
                    if clean_content and len(clean_content) > 50:
                        transcript_text = clean_content
                
                if transcript_text.strip():
                    return transcript_text.strip()
                else:
                    raise ValueError("No text content found using regex")
                    
            except Exception as regex_error:
                raise Exception(f"XML parsing failed: {str(xml_error)} | Regex parsing failed: {str(regex_error)}")
    
    def get_transcript_whisper_alternative(self, video_url):
        """Alternative method using yt-dlp audio extraction for very problematic videos"""
        try:
            # This is a placeholder for a more advanced approach
            # You could integrate with OpenAI Whisper or similar
            return None, "Whisper alternative not implemented yet"
        except Exception as e:
            return None, f"Whisper alternative error: {str(e)}"
    
    def get_transcript(self, video_id, video_url):
        """Get transcript using multiple methods with comprehensive fallbacks"""
        methods = [
            ("YouTube Transcript API", self._get_transcript_youtube_api, video_id),
            ("yt-dlp extraction", self.get_transcript_ytdlp, video_url),
        ]
        
        errors = []
        
        for method_name, method_func, method_input in methods:
            try:
                st.info(f"Trying {method_name}...")
                transcript, error = method_func(method_input)
                
                if transcript and len(transcript.strip()) > 50:  # Lower threshold
                    st.success(f"✅ Successfully extracted transcript using {method_name}")
                    st.info(f"Transcript length: {len(transcript)} characters")
                    return transcript, None
                elif error:
                    errors.append(f"{method_name}: {error}")
                    
            except Exception as e:
                errors.append(f"{method_name}: {str(e)}")
                continue
        
        # If all methods fail, provide detailed debugging info
        debug_info = f"\n**Debug Information:**\n"
        debug_info += f"- Video ID: {video_id}\n"
        debug_info += f"- Video URL: {video_url}\n"
        debug_info += f"- Errors encountered:\n"
        for error in errors:
            debug_info += f"  • {error}\n"
        
        all_errors = " | ".join(errors)
        return None, f"All transcript extraction methods failed: {all_errors}{debug_info}"
    
    def _get_transcript_youtube_api(self, video_id):
        """Original YouTube Transcript API method with better error handling"""
        try:
            # Get available transcript list first
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Debug: Show available transcripts
            available_langs = []
            for transcript in transcript_list:
                available_langs.append(f"{transcript.language} ({'auto' if transcript.is_generated else 'manual'})")
            
            st.info(f"Available transcripts: {', '.join(available_langs) if available_langs else 'None found'}")
            
            # Try to find English transcript first
            transcript = None
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            except:
                # If no English transcript, try any available transcript
                try:
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0]
                        st.warning(f"Using {transcript.language} transcript (English not available)")
                    else:
                        return None, "No transcripts available in transcript list"
                except Exception as inner_e:
                    return None, f"Failed to get transcript list: {str(inner_e)}"
            
            if not transcript:
                return None, "No suitable transcript found"
            
            # Fetch the transcript data
            transcript_data = transcript.fetch()
            
            if not transcript_data:
                return None, "Transcript data is empty"
            
            # Combine transcript segments
            full_transcript = ""
            for segment in transcript_data:
                text = segment.get('text', '')
                if text and text.strip():
                    # Clean the text
                    text = html.unescape(text)
                    full_transcript += text + " "
            
            if not full_transcript.strip():
                return None, "Empty transcript after processing"
                
            return full_transcript.strip(), None
            
        except Exception as e:
            error_msg = str(e).lower()
            if "no transcripts" in error_msg or "transcriptsdisabled" in error_msg:
                return None, "Transcripts are disabled for this video"
            elif "videounavailable" in error_msg:
                return None, "Video is unavailable or private"
            elif "no element found" in error_msg or "not well-formed" in error_msg:
                return None, f"XML parsing error in YouTube API: {str(e)}"
            elif "could not retrieve" in error_msg:
                return None, f"Could not retrieve transcript data: {str(e)}"
            else:
                return None, f"API error: {str(e)}"
    
    def clean_transcript(self, transcript):
        """Clean and preprocess transcript"""
        if not transcript:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', transcript)
        
        # Remove common transcript artifacts
        cleaned = re.sub(r'\[.*?\]', '', cleaned)  # Remove [Music], [Applause], etc.
        cleaned = re.sub(r'\(.*?\)', '', cleaned)  # Remove (inaudible), etc.
        
        # Remove URLs
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        return cleaned.strip()
    
    def cache_transcript(self, video_id, transcript, metadata):
        """Cache transcript and metadata locally"""
        cache_file = os.path.join(self.cache_dir, f"{video_id}.json")
        cache_data = {
            'transcript': transcript,
            'metadata': metadata,
            'cached_at': datetime.now().isoformat()
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Could not cache data: {str(e)}")
    
    def load_cached_transcript(self, video_id):
        """Load cached transcript if available"""
        cache_file = os.path.join(self.cache_dir, f"{video_id}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache is less than 24 hours old
                cached_at = datetime.fromisoformat(cache_data['cached_at'])
                if datetime.now() - cached_at < timedelta(hours=24):
                    return cache_data['transcript'], cache_data['metadata']
            except Exception as e:
                st.warning(f"Could not load cache: {str(e)}")
        
        return None, None
    
    def chunk_transcript(self, transcript, max_chars=30000):
        """Split transcript into chunks for API processing"""
        if len(transcript) <= max_chars:
            return [transcript]
        
        # Split by sentences or paragraphs
        sentences = re.split(r'[.!?]+\s+', transcript)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_chars:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_summary(self, transcript, metadata, summary_type="detailed"):
        """Generate summary using Google Generative AI"""
        try:
            # Prepare the prompt based on summary type
            if summary_type == "detailed":
                prompt = f"""
Please provide a comprehensive and detailed summary of this video transcript. 

**Video Information:**
- Title: {metadata['title']}
- Channel: {metadata['channel']}
- Duration: {metadata['formatted_duration']}
- Upload Date: {metadata['formatted_upload_date']}

**Instructions for Summary:**
1. Create a detailed overview of the main topics and themes
2. Include key points, arguments, and insights presented
3. Organize the content in a logical structure with clear headings
4. Highlight important quotes or statements (if any)
5. Include any actionable advice or conclusions
6. Make it comprehensive but readable

**Transcript:**
{transcript}

Please format the summary with clear sections and bullet points where appropriate.
                """
            elif summary_type == "concise":
                prompt = f"""
Please provide a concise summary of this video transcript.

**Video:** {metadata['title']} by {metadata['channel']}

Focus on:
- Main topic and purpose
- Key points (3-5 bullet points)
- Main conclusion or takeaway

**Transcript:**
{transcript}
                """
            else:  # bullet points
                prompt = f"""
Please create a bullet-point summary of this video transcript.

**Video:** {metadata['title']} by {metadata['channel']}

Create organized bullet points covering:
- Main topics discussed
- Key insights and information
- Important details and facts
- Conclusions or recommendations

**Transcript:**
{transcript}
                """
            
            # Process transcript in chunks if it's too long
            transcript_chunks = self.chunk_transcript(transcript)
            
            if len(transcript_chunks) == 1:
                # Single chunk processing
                response = self.model.generate_content(prompt)
                return response.text
            else:
                # Multi-chunk processing
                summaries = []
                for i, chunk in enumerate(transcript_chunks):
                    chunk_prompt = prompt.replace(transcript, chunk)
                    chunk_prompt += f"\n\n**Note:** This is part {i+1} of {len(transcript_chunks)} of the transcript."
                    
                    response = self.model.generate_content(chunk_prompt)
                    summaries.append(response.text)
                    time.sleep(1)  # Rate limiting
                
                # Combine chunk summaries
                combined_prompt = f"""
Please combine and synthesize these partial summaries into one comprehensive summary:

**Video:** {metadata['title']} by {metadata['channel']}

**Partial Summaries:**
{chr(10).join([f"Part {i+1}: {summary}" for i, summary in enumerate(summaries)])}

Create a unified, well-organized final summary that combines all the information coherently.
                """
                
                final_response = self.model.generate_content(combined_prompt)
                return final_response.text
                
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("🎥 YouTube Video Summarizer")
    st.markdown("Enter a YouTube video URL to get an AI-powered detailed summary using Google's Gemini AI.")
    
    # Initialize summarizer
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = YouTubeSummarizer()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        summary_type = st.selectbox(
            "Summary Type:",
            ["detailed", "concise", "bullet_points"],
            format_func=lambda x: {
                "detailed": "📄 Detailed Summary",
                "concise": "📋 Concise Summary", 
                "bullet_points": "🔸 Bullet Points"
            }[x]
        )
        
        show_transcript = st.checkbox("Show Full Transcript", value=False)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Works with public YouTube videos
        - Requires video to have transcripts/captions
        - Processing may take 30-60 seconds
        - Longer videos are processed in chunks
        """)
    
    # Main input
    video_url = st.text_input(
        "YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any YouTube video URL here"
    )
    
    if st.button("🎯 Generate Summary", type="primary"):
        if not video_url:
            st.error("Please enter a YouTube video URL.")
            return
        
        # Extract video ID
        video_id = st.session_state.summarizer.extract_video_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Check cache
            status_text.text("🔍 Checking cache...")
            progress_bar.progress(10)
            
            cached_transcript, cached_metadata = st.session_state.summarizer.load_cached_transcript(video_id)
            
            if cached_transcript and cached_metadata:
                st.success("📋 Found cached data!")
                transcript = cached_transcript
                metadata = cached_metadata
                progress_bar.progress(50)
            else:
                # Step 2: Extract metadata
                status_text.text("📊 Extracting video metadata...")
                progress_bar.progress(20)
                
                metadata = st.session_state.summarizer.get_video_metadata(video_url)
                if not metadata:
                    st.error("Could not extract video metadata. Please check if the video is public and accessible.")
                    return
                
                # Step 3: Get transcript
                status_text.text("📝 Extracting transcript...")
                progress_bar.progress(30)
                
                raw_transcript, error = st.session_state.summarizer.get_transcript(video_id, video_url)
                if error:
                    st.error(f"Could not get transcript: {error}")
                    st.info("💡 **Troubleshooting Tips:**")
                    st.markdown("""
                    - Make sure the video has captions/subtitles enabled
                    - Try with a different YouTube video that has captions
                    - Some videos may have restricted access to transcripts
                    - Check if the video is public and accessible
                    """)
                    return
                
                # Step 4: Clean transcript
                status_text.text("🧹 Cleaning transcript...")
                progress_bar.progress(40)
                
                transcript = st.session_state.summarizer.clean_transcript(raw_transcript)
                
                # Cache the data
                st.session_state.summarizer.cache_transcript(video_id, transcript, metadata)
                progress_bar.progress(50)
            
            # Step 5: Generate summary
            status_text.text("🤖 Generating AI summary...")
            progress_bar.progress(60)
            
            summary = st.session_state.summarizer.generate_summary(transcript, metadata, summary_type)
            
            progress_bar.progress(100)
            status_text.text("✅ Summary generated successfully!")
            
            if summary:
                # Display results
                st.markdown("---")
                
                # Video metadata
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📺 Video Information")
                    st.write(f"**Title:** {metadata['title']}")
                    st.write(f"**Channel:** {metadata['channel']}")
                    st.write(f"**Duration:** {metadata['formatted_duration']}")
                    st.write(f"**Upload Date:** {metadata['formatted_upload_date']}")
                    if metadata['view_count']:
                        st.write(f"**Views:** {metadata['view_count']:,}")
                
                with col2:
                    st.subheader("📊 Stats")
                    st.metric("Transcript Length", f"{len(transcript):,} chars")
                    st.metric("Word Count", f"{len(transcript.split()):,} words")
                
                # Summary
                st.subheader("📋 AI-Generated Summary")
                st.markdown(summary)
                
                # Transcript (optional)
                if show_transcript:
                    st.subheader("📜 Full Transcript")
                    with st.expander("Click to view full transcript", expanded=False):
                        st.text_area("Transcript:", transcript, height=300)
                
                # Download options
                st.subheader("💾 Download")
                col1, col2 = st.columns(2)
                
                with col1:
                    summary_text = f"# {metadata['title']}\n\n"
                    summary_text += f"**Channel:** {metadata['channel']}\n"
                    summary_text += f"**Duration:** {metadata['formatted_duration']}\n"
                    summary_text += f"**Upload Date:** {metadata['formatted_upload_date']}\n\n"
                    summary_text += "## Summary\n\n"
                    summary_text += summary
                    
                    st.download_button(
                        "📄 Download Summary",
                        summary_text,
                        file_name=f"{metadata['title'][:50]}_summary.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    if show_transcript:
                        st.download_button(
                            "📜 Download Transcript", 
                            transcript,
                            file_name=f"{metadata['title'][:50]}_transcript.txt",
                            mime="text/plain"
                        )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()

