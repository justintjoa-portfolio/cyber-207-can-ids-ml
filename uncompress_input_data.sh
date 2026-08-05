cd original_input_data

for archive in *.tar.gz *.tgz; do
    [ -e "$archive" ] || continue
    tar -xzf "$archive"
done