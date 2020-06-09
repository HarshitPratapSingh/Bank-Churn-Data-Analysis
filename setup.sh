mkdir -p ~/.streamlit/
echo “\
[general]\n\
email = \”4harshitsingh@gmail.com\”\n\
“ > ~/.streamlit/credentials.toml
echo “\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
“ > ~/.streamlit/config.toml