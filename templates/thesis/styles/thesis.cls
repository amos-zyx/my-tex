\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{styles/thesis}[2/23/2023 v2.0 template of thesis or course report for self-using]


%% some beautiful color
\RequirePackage{xcolor}
\definecolor{slateBlue}{rgb}{0.17,0.22,0.34}
\definecolor{darkCyan}{RGB}{0,139,139}
\definecolor{skyBlue}{RGB}{30,144,255}
\definecolor{CornflowerBlue}{RGB}{100,149,237}

% title color -> only change color of title, secion, subsection, etc
\definecolor{titlecolor}{RGB}{0,0,0} % 黑色
% \definecolor{titlecolor}{RGB}{179,27,27} % 深红色


%% documentclass = ctex
\LoadClass[UTF8, a4paper, fontset=none, 10pt]{ctexart}
% \LoadClass[UTF8, a4paper, fontset=none, 10pt]{ctexrep}
\ctexset{
  today = small,
  % chapter = { % for ctexrep
  %   format += \raggedright\sffamily\color{titlecolor},
  %   % beforeskip = -30pt,
  %   name = {Chapter\hspace{.5em}},
  %   number = {\arabic{chapter}}
  % },
  section = {
    format += \raggedright\sffamily\color{titlecolor},
    % name = {\S}, % "\S 1"
  },
  subsection/format += \sffamily\color{titlecolor},
  subsubsection/format += \sffamily\color{titlecolor},
}
\renewcommand{\abstractname}{\Large\bfseries\sffamily\color{titlecolor}摘要}


%% font
\RequirePackage{fontspec}

% English
\setmonofont[
    ItalicFont = Fira Code Light
]{Fira Code}

% Chinese
\setCJKmainfont[
    ItalicFont = 方正新楷体简体,
    BoldItalicFont = 方正盛世楷书简体_粗
]{Source Han Serif CN}

\setCJKsansfont[
    ItalicFont = 方正新楷体简体,
    BoldItalicFont = 方正盛世楷书简体_粗
]{Source Han Serif CN}

\setCJKmonofont[
    BoldFont = Sarasa Mono Slab SC,
]{Sarasa Mono Slab SC Light}


%% 页面布局样式
% 页边距
\RequirePackage[a4paper, hmargin=20mm, vmargin={25mm,20mm}]{geometry}

% 页眉, 页脚
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\pagestyle{fancy}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage {} / \pageref*{LastPage}}
\cfoot{}
\renewcommand{\headrulewidth}{1pt}


%% other settings
% hyperlink
\RequirePackage[colorlinks, linkcolor=darkCyan, anchorcolor=darkCyan, citecolor=darkCyan, bookmarksnumbered=true]{hyperref}

% appendix
\RequirePackage[title, titletoc]{appendix}

% references
\PassOptionsToPackage{
  backend=biber,
  style=gb7714-2015ms,
  gbstyle=false,
  gbalign=gb7714-2015,
  gbnamefmt=lowercase,
  doi=true,
  url=false,
}{biblatex}

% 限制浮动体位置
\RequirePackage[above, section]{placeins}  % provide \FloatBarrier (禁止浮动体越过)

% define \addinputpath
\makeatletter
\providecommand*\input@path{}
\newcommand\addinputpath[1]{
  \expandafter\def\expandafter\input@path
  \expandafter{\input@path{#1}}}
\makeatother
