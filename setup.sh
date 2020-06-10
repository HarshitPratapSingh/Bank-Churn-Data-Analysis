mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"4harshitsingh@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = false\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml