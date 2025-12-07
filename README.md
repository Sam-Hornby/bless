Binary Less, a terminal viewer for binary files and tabular data. Currently supports IPC, Parquet, and CSV. 

To build 
```
git clone https://github.com/Sam-Hornby/bless.git
cd bless
cargo build --release  
```

This will make a binary at `bless/target/release/bless`. Can then run this on your files eg 
```
target/release/bless my_data.ipc
```

Supports searching like `less` with `/`. Can also execute SQL on the tabular data with `:` and using the table name as `current`. 
