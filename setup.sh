mkdir -p ~/.streamlit/

echo "\
[general]\n\
<<<<<<< HEAD
email = \"4harshitsingh@gmail.com\"\n\
=======
email = \"your-email@domain.com\"\n\
>>>>>>> 0c66e970dbf4b2f391ebae6e3113a6194451e01a
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
<<<<<<< HEAD
enableCORS=false\n\
=======
enableCORS=true\n\
>>>>>>> 0c66e970dbf4b2f391ebae6e3113a6194451e01a
port = $PORT\n\
" > ~/.streamlit/config.toml