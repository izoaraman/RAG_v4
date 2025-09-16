import gradio as gr
from typing import List, Dict, Any

class CorporateUI:
    """
    Corporate-style UI components with updated brand palette
    """
    
    # Updated Brand Palette
    BRAND_COLORS = {
        'violet': '#342D8C',      # Core headers, CTA hover, icon tints
        'opal': '#1CCFC9',        # Primary buttons, active elements, key links
        'navy': '#131838',        # Body copy, footers, captions
        'forest': '#38862E',      # Success badges / "approved" states
        'sea': '#008098',         # Secondary buttons, tabs, info banners
        'gold': '#D5A70B',        # Warning icons, rating stars
        'raspberry': '#BC204B',   # Error / destructive buttons
        'charcoal': '#373737',    # Neutral dividers, secondary text
        'lime': '#97C76A',        # Positive trend lines, data-viz accents
        'sky': '#77B5DD',         # Chart series, subtle backgrounds
        'wheat': '#FDC94D',       # Highlight cards, KPI call-outs
        'rose': '#E85D72',        # Accent badges, infographic highlights
        'white': '#FFFFFF',
        'off_white': '#F8F9FA',
        'light_gray': '#E9ECEF',
        'border_gray': '#DEE2E6'
    }
    
    # Typography
    FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
    
    @staticmethod
    def get_corporate_css() -> str:
        """Get Corporate-style CSS with updated brand colors"""
        return f"""
        /* Global Styles */
        * {{
            font-family: {CorporateUI.FONT_FAMILY};
            box-sizing: border-box;
        }}
        
        /* Main Container */
        .gradio-container {{
            background-color: {CorporateUI.BRAND_COLORS['off_white']};
            max-width: 1400px !important;
            margin: 0 auto;
        }}
        
        /* Chat Container */
        .chat-container {{
            background: {CorporateUI.BRAND_COLORS['white']};
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            border: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
            overflow: hidden;
        }}
        
        /* Answer Section */
        .answer-section {{
            background: {CorporateUI.BRAND_COLORS['white']};
            border: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        }}
        
        .answer-header {{
            display: flex;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid {CorporateUI.BRAND_COLORS['opal']};
        }}
        
        .answer-icon {{
            width: 24px;
            height: 24px;
            background: {CorporateUI.BRAND_COLORS['violet']};
            border-radius: 6px;
            margin-right: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {CorporateUI.BRAND_COLORS['white']};
            font-weight: 600;
            font-size: 14px;
        }}
        
        .answer-title {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {CorporateUI.BRAND_COLORS['violet']};
            margin: 0;
        }}
        
        .answer-content {{
            line-height: 1.6;
            color: {CorporateUI.BRAND_COLORS['navy']};
            font-size: 1rem;
        }}
        
        .answer-content h1, .answer-content h2, .answer-content h3,
        .answer-content h4, .answer-content h5, .answer-content h6 {{
            color: {CorporateUI.BRAND_COLORS['violet']};
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        
        .answer-content a {{
            color: {CorporateUI.BRAND_COLORS['opal']};
            text-decoration: none;
        }}
        
        .answer-content a:hover {{
            text-decoration: underline;
        }}
        
        /* Sources Section */
        .sources-section {{
            background: {CorporateUI.BRAND_COLORS['light_gray']};
            border: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }}
        
        .sources-header {{
            display: flex;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
        }}
        
        .sources-icon {{
            width: 20px;
            height: 20px;
            background: {CorporateUI.BRAND_COLORS['sea']};
            border-radius: 4px;
            margin-right: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {CorporateUI.BRAND_COLORS['white']};
            font-size: 12px;
            font-weight: 600;
        }}
        
        .sources-title {{
            font-size: 1rem;
            font-weight: 600;
            color: {CorporateUI.BRAND_COLORS['navy']};
            margin: 0;
        }}
        
        .sources-count {{
            background: {CorporateUI.BRAND_COLORS['opal']};
            color: {CorporateUI.BRAND_COLORS['white']};
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: auto;
            font-weight: 500;
        }}
        
        .source-item {{
            background: {CorporateUI.BRAND_COLORS['white']};
            border: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        
        .source-item:hover {{
            border-color: {CorporateUI.BRAND_COLORS['opal']};
            box-shadow: 0 2px 8px rgba(28, 207, 201, 0.15);
            transform: translateY(-1px);
        }}
        
        .source-item-title {{
            font-weight: 600;
            color: {CorporateUI.BRAND_COLORS['violet']};
            margin-bottom: 6px;
            font-size: 0.95rem;
        }}
        
        .source-item-meta {{
            color: {CorporateUI.BRAND_COLORS['charcoal']};
            font-size: 0.85rem;
            margin-bottom: 8px;
        }}
        
        .source-item-content {{
            color: {CorporateUI.BRAND_COLORS['navy']};
            font-size: 0.9rem;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        
        /* Steps Section */
        .steps-section {{
            background: {CorporateUI.BRAND_COLORS['white']};
            border: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }}
        
        .steps-header {{
            display: flex;
            align-items: center;
            margin-bottom: 16px;
            padding-bottom: 10px;
            border-bottom: 1px solid {CorporateUI.BRAND_COLORS['border_gray']};
        }}
        
        .steps-icon {{
            width: 20px;
            height: 20px;
            background: {CorporateUI.BRAND_COLORS['forest']};
            border-radius: 4px;
            margin-right: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {CorporateUI.BRAND_COLORS['white']};
            font-size: 12px;
            font-weight: 600;
        }}
        
        .step-item {{
            padding: 12px 0;
            border-bottom: 1px solid {CorporateUI.BRAND_COLORS['light_gray']};
        }}
        
        .step-item:last-child {{
            border-bottom: none;
        }}
        
        .step-number {{
            display: inline-block;
            width: 24px;
            height: 24px;
            background: {CorporateUI.BRAND_COLORS['opal']};
            color: {CorporateUI.BRAND_COLORS['white']};
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 12px;
        }}
        
        .step-content {{
            display: inline;
            color: {CorporateUI.BRAND_COLORS['navy']};
        }}
        
        /* Buttons */
        .btn-primary {{
            background: {CorporateUI.BRAND_COLORS['opal']};
            border: none;
            color: {CorporateUI.BRAND_COLORS['white']};
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
            cursor: pointer;
            font-size: 1rem;
        }}
        
        .btn-primary:hover {{
            background: linear-gradient(135deg, {CorporateUI.BRAND_COLORS['opal']} 0%, {CorporateUI.BRAND_COLORS['violet']} 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(28, 207, 201, 0.3);
        }}
        
        /* Gradio Primary Button Override */
        button.primary {{
            background: {CorporateUI.BRAND_COLORS['opal']} !important;
            border: none !important;
            color: {CorporateUI.BRAND_COLORS['white']} !important;
        }}
        
        button.primary:hover {{
            background: {CorporateUI.BRAND_COLORS['violet']} !important;
            opacity: 0.9 !important;
        }}
        
        .btn-secondary {{
            background: transparent;
            border: 2px solid {CorporateUI.BRAND_COLORS['sea']};
            color: {CorporateUI.BRAND_COLORS['sea']};
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        
        .btn-secondary:hover {{
            background: {CorporateUI.BRAND_COLORS['sea']};
            color: {CorporateUI.BRAND_COLORS['white']};
        }}
        
        /* Input Styles */
        .corporate-input {{
            border: 2px solid {CorporateUI.BRAND_COLORS['border_gray']};
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 1rem;
            transition: all 0.2s ease;
            background: {CorporateUI.BRAND_COLORS['white']};
            color: {CorporateUI.BRAND_COLORS['navy']};
        }}
        
        .corporate-input:focus {{
            border-color: {CorporateUI.BRAND_COLORS['opal']};
            box-shadow: 0 0 0 3px rgba(28, 207, 201, 0.15);
            outline: none;
        }}
        
        /* Success, Warning, Error States */
        .success-badge {{
            background: {CorporateUI.BRAND_COLORS['forest']};
            color: {CorporateUI.BRAND_COLORS['white']};
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .warning-badge {{
            background: {CorporateUI.BRAND_COLORS['gold']};
            color: {CorporateUI.BRAND_COLORS['navy']};
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        .error-badge {{
            background: {CorporateUI.BRAND_COLORS['raspberry']};
            color: {CorporateUI.BRAND_COLORS['white']};
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        
        /* Layout */
        .main-layout {{
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 24px;
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        
        .sidebar {{
            background: {CorporateUI.BRAND_COLORS['white']};
            border-radius: 12px;
            padding: 20px;
            height: fit-content;
            position: sticky;
            top: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }}
        
        .main-content {{
            min-height: 80vh;
        }}
        
        /* Responsive Design */
        @media (max-width: 1024px) {{
            .main-layout {{
                grid-template-columns: 1fr;
                gap: 16px;
                padding: 16px;
            }}
            
            .sidebar {{
                position: static;
                order: 2;
            }}
            
            .main-content {{
                order: 1;
            }}
        }}
        
        @media (max-width: 768px) {{
            .answer-section, .sources-section, .steps-section {{
                padding: 16px;
                margin: 12px 0;
            }}
            
            .btn-primary, .btn-secondary {{
                padding: 10px 16px;
                font-size: 0.9rem;
            }}
        }}
        
        /* Animation Classes */
        .fade-in {{
            animation: fadeIn 0.3s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Accessibility */
        .sr-only {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }}
        
        /* Focus states */
        .btn-primary:focus,
        .btn-secondary:focus,
        .corporate-input:focus,
        .source-item:focus {{
            outline: 3px solid {CorporateUI.BRAND_COLORS['opal']};
            outline-offset: 2px;
        }}
        
        /* High contrast mode */
        @media (prefers-contrast: high) {{
            .answer-section, .sources-section, .steps-section {{
                border-width: 2px;
            }}
        }}
        
        /* Reduced motion */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation: none !important;
                transition: none !important;
            }}
        }}
        """
    
    @staticmethod
    def create_answer_component(content: str) -> str:
        """Create Corporate-style answer section"""
        return f"""
        <div class="answer-section fade-in" role="article" aria-labelledby="answer-heading">
            <div class="answer-header">
                <div class="answer-icon" aria-hidden="true">A</div>
                <h2 class="answer-title" id="answer-heading">Answer</h2>
            </div>
            <div class="answer-content" role="main">
                {content}
            </div>
        </div>
        """
    
    @staticmethod
    def create_sources_component(sources: List[Dict[str, Any]]) -> str:
        """Create Corporate-style sources section"""
        if not sources:
            return ""
        
        source_items = ""
        for i, source in enumerate(sources[:10]):
            title = source.get('title', f'Document {i+1}')
            page = source.get('page', 'N/A')
            filename = source.get('filename', 'Unknown')
            content = source.get('content', '')[:200]
            source_url = source.get('source_url', '#')
            
            source_items += f"""
            <div class="source-item" tabindex="0" role="article" aria-labelledby="source-{i}">
                <div class="source-item-title" id="source-{i}">{title}</div>
                <div class="source-item-meta">Page {page} • {filename}</div>
                <div class="source-item-content">{content}...</div>
                <a href="{source_url}" class="sr-only">View full document</a>
            </div>
            """
        
        return f"""
        <div class="sources-section fade-in" role="complementary" aria-labelledby="sources-heading">
            <div class="sources-header">
                <div class="sources-icon" aria-hidden="true">S</div>
                <h3 class="sources-title" id="sources-heading">Sources</h3>
                <span class="sources-count" aria-label="{len(sources)} sources">{len(sources)}</span>
            </div>
            <div role="list">
                {source_items}
            </div>
        </div>
        """
    
    @staticmethod
    def create_steps_component(steps: List[str]) -> str:
        """Create Corporate-style steps section"""
        if not steps:
            return ""
        
        step_items = ""
        for i, step in enumerate(steps, 1):
            step_items += f"""
            <div class="step-item">
                <span class="step-number">{i}</span>
                <span class="step-content">{step}</span>
            </div>
            """
        
        return f"""
        <div class="steps-section fade-in" role="article" aria-labelledby="steps-heading">
            <div class="steps-header">
                <div class="steps-icon" aria-hidden="true">✓</div>
                <h3 class="sources-title" id="steps-heading">Steps</h3>
            </div>
            <div role="list">
                {step_items}
            </div>
        </div>
        """
    
    @staticmethod
    def get_theme_tokens() -> Dict[str, str]:
        """Export theme tokens for external use"""
        return {
            "colors": CorporateUI.BRAND_COLORS,
            "fontFamily": CorporateUI.FONT_FAMILY,
            "spacing": {
                "xs": "4px",
                "sm": "8px",
                "md": "16px",
                "lg": "24px",
                "xl": "32px"
            },
            "borderRadius": {
                "sm": "4px",
                "md": "6px",
                "lg": "8px",
                "xl": "12px"
            }
        }