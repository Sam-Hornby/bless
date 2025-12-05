#![deny(warnings)]
use crossterm::event::{self, Event, KeyEvent, KeyCode};
use ratatui::{Frame, DefaultTerminal};
use ratatui::widgets::{Table, TableState, Row, Paragraph, Scrollbar, ScrollbarState, ScrollbarOrientation};
use ratatui::text::{Line, Text};
use ratatui::layout::{Constraint, Rect, Margin};
use ratatui::style::Color;
use clap::{Parser};
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;
use polars::prelude::{CsvReadOptions, PolarsError, SerReader, DataType};
use polars_lazy::prelude::{LazyFrame, len, col, any_horizontal, Expr, lit};
use polars_lazy::frame::IntoLazy;
use std::io::Error as IOError;
use std::io::ErrorKind;
use env_logger;
use log::{debug, LevelFilter};
use chrono::Local;
use std::cmp;
use polars_io::ipc::IpcReader;
use polars_io::ipc::IpcStreamReader;
use polars_io::prelude::ParquetReader;
use polars_sql::SQLContext;

// Structs, Enums, Data
// -------------------------------------------------------------------------------

struct TuiState {
    scroll_bar_state: ScrollbarState,
    table_state: TableState,
}

#[derive(Debug)]
struct Coordinate {
    row: i64,
    column: i64,
}

struct DataFrameModel {
    current_frame: LazyFrame,
    current_position: Coordinate,
    total_num_rows: i64,
    total_num_columns: i64,
}

struct Hdf5Model {}

enum TableModel {
    Csv(DataFrameModel),
    Ipc(DataFrameModel),
    Parquet(DataFrameModel),
    IpcStream(DataFrameModel),
    Hdf5(Hdf5Model),
}

#[derive(Debug, Clone)]
enum SearchModel {
    SearchCreation(String),
    SearchInProgress(String),
}

#[derive(Debug)]
enum UserModel {
    DefaultMode,
    SearchMode(SearchModel),
    CommandMode(String),
}

struct DataModel {
    user_model: UserModel,
    table_model: TableModel,
}

#[derive(Debug)]
enum Action {
    // Control actions
    Quit,
    Noop,
    // Scroll actions
    ScrollDown,
    ScrollUp,
    ScrollLeft,
    ScrollRight,
    // Search actions
    SearchBegin,
    SearchAddKey(char),
    SearchRemoveKey,
    SearchUp,
    SearchDown,
    SearchEnd,
    // Command actions
    CommandBegin,
    CommandAddKey(char),
    CommandRemoveKey,
    CommandExecute(String),
    CommandEnd,
}

struct ScrollIncrement {
    vertical_increment: i64,
    horizontal_increment: i64,
}

fn convert_polars_error(error: PolarsError) -> String {
    format!("Polars Error: {}", error)
}

fn convert_to_display_error(error: std::io::Error) -> String {
    format!("IO Error: {}", error)
}

fn create_data_frame_model(whole_frame: LazyFrame) -> Result<DataFrameModel, String> {
    let total_rows = get_lazy_frame_total_rows(whole_frame.clone());
    let schema = whole_frame.clone().collect_schema().map_err(convert_polars_error)?;
    debug!("Loaded Schema {:?}", schema);
    debug!("Shape {} / {}", schema.len(), total_rows);
    Ok(DataFrameModel {
        current_frame: whole_frame,
        current_position: Coordinate{row: 0, column:0},
        total_num_rows: total_rows as i64,
        total_num_columns: schema.len() as i64,
    })
}

fn create_csv_model(file: &PathBuf) -> Result<DataFrameModel, String> {
    let csv_reader = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(file.clone()))
            .map_err(convert_polars_error)?;
    let df = csv_reader.finish().map_err(convert_polars_error)?;
    create_data_frame_model(df.lazy())
}

fn create_ipc_model(file: &PathBuf) -> Result<DataFrameModel, String> {
    let stream = File::open(file).map_err(convert_to_display_error)?;
    let df = IpcReader::new(stream).finish().map_err(convert_polars_error)?;
    create_data_frame_model(df.lazy())
}

fn create_parquet_model(file: &PathBuf) -> Result<DataFrameModel, String> {
    let stream = File::open(file).map_err(convert_to_display_error)?;
    let df = ParquetReader::new(stream).finish().map_err(convert_polars_error)?;
    create_data_frame_model(df.lazy())
}

fn create_ipc_stream_model(file: &PathBuf) -> Result<DataFrameModel, String> {
    let stream = File::open(file).map_err(convert_to_display_error)?;
    let df = IpcStreamReader::new(stream).finish().map_err(convert_polars_error)?;
    create_data_frame_model(df.lazy())
}

#[derive(Parser)]
struct Cli {
    input_file: PathBuf,
    #[arg(long)]
    file_type: Option<String>,
    #[arg(long)]
    log_dir: Option<PathBuf>,
}

fn create_table_model_from_unknowwn_file_type(input_file: &PathBuf) -> Result<TableModel, String> {
    let mut error_string : String = "Couldn't load via any known file types! Errors were:".to_owned();
    match create_ipc_model(input_file) {
        Ok(model) => { return Ok(TableModel::Ipc(model)); },
        Err(e) => {
            error_string.push_str("\n  ");
            error_string.push_str(&e);
        }
    }

    match create_parquet_model(input_file) {
        Ok(model) => { return Ok(TableModel::Parquet(model)); },
        Err(e) => {
            error_string.push_str("\n  ");
            error_string.push_str(&e);
        }
    }

    match create_ipc_stream_model(input_file) {
        Ok(model) => { return Ok(TableModel::IpcStream(model)); },
        Err(e) => {
            error_string.push_str("\n  ");
            error_string.push_str(&e);
        }
    }

    match create_csv_model(input_file) {
        Ok(model) => { return Ok(TableModel::Csv(model)); },
        Err(e) => {
            error_string.push_str("\n  ");
            error_string.push_str(&e);
        }
    }

    error_string.push_str("\n");
    Err(error_string)
}

fn create_table_model(input_file: &PathBuf, file_type: &Option<String>) -> Result<TableModel, String> {
    match file_type {
        None => create_table_model_from_unknowwn_file_type(input_file),
        Some(file_type) => match file_type.to_lowercase().as_str() {
            "csv" => Ok(TableModel::Csv(create_csv_model(input_file)?)),
            "ipc" => Ok(TableModel::Ipc(create_ipc_model(input_file)?)),
            "parquet" => Ok(TableModel::Parquet(create_parquet_model(input_file)?)),
            "ipc_stream" => Ok(TableModel::IpcStream(create_ipc_stream_model(input_file)?)),
            "hdf5" => Ok(TableModel::Hdf5(Hdf5Model{})),
            _ => Err("Unsupported file type".to_string()),
        }
    }
}

fn create_model(cli: &Cli) -> Result<DataModel, String> {
    let table_model = create_table_model(&cli.input_file, &cli.file_type)?;
    Ok(DataModel{
        user_model: UserModel::DefaultMode,
        table_model: table_model,
    })
}

fn data_frame_scroll(model: &mut DataFrameModel, increment: ScrollIncrement) -> Result<(), String> {
    model.current_position.row += increment.vertical_increment;
    model.current_position.row = cmp::max(model.current_position.row, 0);
    model.current_position.row = cmp::min(model.current_position.row, model.total_num_rows - 1);

    model.current_position.column += increment.horizontal_increment;
    model.current_position.column = cmp::max(model.current_position.column, 0);
    model.current_position.column = cmp::min(model.current_position.column, model.total_num_columns - 1);

    Ok(())
}

fn scroll_update_state(model: &mut DataModel, increment: ScrollIncrement) -> Result<(), String> {
    match &mut model.table_model {
        TableModel::Csv(csv) => data_frame_scroll(csv, increment),
        TableModel::Ipc(ipc) => data_frame_scroll(ipc, increment),
        TableModel::Parquet(parquet) => data_frame_scroll(parquet, increment),
        TableModel::IpcStream(ipc) => data_frame_scroll(ipc, increment),
        TableModel::Hdf5(_) => Err(format!("Scroll not implemented for hdf5")),
    }
}

fn data_frame_get_search_index(model: &DataFrameModel, search_string: &str, _action: &Action) -> Result<Option<Coordinate>, String> {
    let start = model.current_position.row;
    let end = model.total_num_rows;
    let schema = model.current_frame.clone().collect_schema().map_err(convert_polars_error)?;
    let filter_expr = any_horizontal(
        schema.iter_names().map(|name| {
            col(name.as_str()).str().contains(lit(search_string), true)
        }).collect::<Vec<Expr>>()
    ).map_err(convert_polars_error)?;

    let found_in_rows = model.current_frame.clone().slice(start, (end - start) as u32)
        .cast_all(DataType::String, false)
        .with_row_index("bless_row_index", Some(start as u32))
        .filter(filter_expr)
        .slice(0, 2)
        .collect()
        .map_err(convert_polars_error)?;
    if found_in_rows.iter().len() == 0 {
        return Ok(None);
    }
    let bless_column = found_in_rows.column("bless_row_index").map_err(convert_polars_error)?;
    let bless_column_as_array = bless_column.u32().map_err(convert_polars_error)?;
    let first_row_index = bless_column_as_array.iter().min().ok_or("Should of at least found starting row")?.ok_or("Internal row was null")?;
    let second_row_index = bless_column_as_array.iter().max().ok_or("Should of at least found starting row")?.ok_or("Internal row was null")?;

    let first_row_result  = found_in_rows.get_row(0).map_err(convert_polars_error)?.0.iter()
        .enumerate()
        .filter(|x| {
            let value = x.1.get_str();
            match value {
                None => false,
                Some(v) => v.contains(search_string),
            }
        }).filter(
            |x| ((x.0 as i64) - 1) > model.current_position.column
        ).map(
            |x| Coordinate{row: first_row_index as i64, column: (x.0 as i64) - 1}
        ).next();

    if second_row_index == first_row_index {
        return Ok(first_row_result);
    }

    let second_row_result = found_in_rows.get_row(1).map_err(convert_polars_error)?.0.iter()
        .enumerate()
        .filter(|x| {
            let value = x.1.get_str();
            match value {
                None => false,
                Some(v) => v.contains(search_string),
            }
        }).map(|x| Coordinate{row: second_row_index as i64, column: (x.0 as i64) - 1})
        .next();
    debug!("{:?} and {:?}", first_row_result, second_row_result);
    match first_row_result {
        Some(coord) => Ok(Some(coord)),
        None => match second_row_result {
            None => Ok(None),
            Some(second_coord) => Ok(Some(second_coord)),
        }
    }
}


fn data_frame_search(model: &mut DataFrameModel, search_string: &str, action: &Action) -> Result<(), String> {
    let postion_opt = data_frame_get_search_index(model, search_string, action)?;
    debug!("Search for {} returned {:?}", search_string, postion_opt);
    if let Some(position) = postion_opt {
        model.current_position = position;
    }
    Ok(())
 }

 fn execute_search(model: &mut TableModel, search_string: &str, action: &Action) -> Result<(), String> {
    match model {
        TableModel::Csv(csv) => data_frame_search(csv, search_string, action),
        TableModel::Ipc(ipc) => data_frame_search(ipc, search_string, action),
        TableModel::Parquet(parquet) => data_frame_search(parquet, search_string, action),
        TableModel::IpcStream(ipc) => data_frame_search(ipc, search_string, action),
        TableModel::Hdf5(_) => Err(format!("Search not implemented for hdf5")),
    }
 }

 fn execute_command_on_data_frame(model: &mut DataFrameModel, command_string: &str) -> Result<(), String> {
    let mut context = SQLContext::new();
    context.register("current", model.current_frame.clone());
    let sql_df = context.execute(command_string).map_err(convert_polars_error)?;
    *model = create_data_frame_model(sql_df)?;
    Ok(())
 }

 fn execute_command(model: &mut DataModel, command_string: &str) -> Result<(), String> {
    match &mut model.table_model {
        TableModel::Csv(csv) => execute_command_on_data_frame(csv, command_string),
        TableModel::Ipc(ipc) => execute_command_on_data_frame(ipc, command_string),
        TableModel::Parquet(parquet) => execute_command_on_data_frame(parquet, command_string),
        TableModel::IpcStream(ipc) => execute_command_on_data_frame(ipc, command_string),
        TableModel::Hdf5(_) => Err(format!("Command Execution not supported for hdf5")),
    }?;
    model.user_model = UserModel::DefaultMode;
    Ok(())
}

fn update_state(model: &mut DataModel, action: &Action) -> Result<(), String> {
    return match action {
        Action::Quit => Ok(()),
        Action::Noop => Ok(()),
        Action::ScrollDown => scroll_update_state(model, ScrollIncrement{vertical_increment: 1, horizontal_increment: 0}),
        Action::ScrollUp => scroll_update_state(model, ScrollIncrement{vertical_increment: -1, horizontal_increment: 0}),
        Action::ScrollLeft => scroll_update_state(model, ScrollIncrement{vertical_increment: 0, horizontal_increment: -1}),
        Action::ScrollRight => scroll_update_state(model, ScrollIncrement{vertical_increment: 0, horizontal_increment: 1}),
        Action::SearchBegin => {
            model.user_model = UserModel::SearchMode(SearchModel::SearchCreation(String::new()));
            Ok(())
        },
        Action::SearchAddKey(c) => {
            if let UserModel::SearchMode(ref mut search_model) = model.user_model {
                if let SearchModel::SearchCreation(search_string) = search_model {
                    search_string.push(*c);
                    return Ok(());
                }
                return Err(format!("Invalid state for search add key"));
            }
            return Err(format!("Invalid mode for search add key"));
        },
        Action::SearchRemoveKey => {
            if let UserModel::SearchMode(ref mut search_model) = model.user_model {
                if let SearchModel::SearchCreation(search_string) = search_model {
                    search_string.pop();
                    return Ok(());
                }
                return Err(format!("Invalid state for search remove key"));
            }
            return Err(format!("Invalid mode for search remove key"));
        },
        Action::SearchUp | Action::SearchDown => {
            if let UserModel::SearchMode(ref mut search_model) = model.user_model {
                if let SearchModel::SearchCreation(search_string) = search_model.clone() {
                    *search_model = SearchModel::SearchInProgress(search_string);
                }
                if let SearchModel::SearchInProgress(search_string) = search_model {
                    return execute_search(&mut model.table_model, search_string, &action);
                }
                return Err(format!("Invalid state for search action"));
            }
            return Err(format!("Invalid mode for search action"));
        },
        Action::SearchEnd => {
            model.user_model = UserModel::DefaultMode;
            Ok(())
        },
        Action::CommandBegin => {
            model.user_model = UserModel::CommandMode(String::new());
            Ok(())
        },
        Action::CommandAddKey(c) => {
            if let UserModel::CommandMode(ref mut command_string) = model.user_model {
                command_string.push(*c);
                return Ok(());
            }
            Err(format!("Invalid mode for CommandAddKey"))
        },
        Action::CommandRemoveKey => {
            if let UserModel::CommandMode(ref mut command_string) = model.user_model {
                command_string.pop();
                return Ok(());
            }
            Err(format!("Invalid model for CommandRemoveKey"))
        },
        Action::CommandExecute(command) => execute_command(model, &command),
        Action::CommandEnd => {
            model.user_model = UserModel::DefaultMode;
            Ok(())
        },

    }
}


fn get_default_mode_action(key_event: &KeyEvent) -> Result<Action, String> {
    match key_event.code {
        KeyCode::Char(c) => {
            if c == 'q' {
                return Ok(Action::Quit);
            }
            if c == '/' {
                return Ok(Action::SearchBegin);
            }
            if c == ':' {
                return Ok(Action::CommandBegin);
            }
            Ok(Action::Noop)
        }
        KeyCode::Down => Ok(Action::ScrollDown),
        KeyCode::Up => Ok(Action::ScrollUp),
        KeyCode::Left => Ok(Action::ScrollLeft),
        KeyCode::Right => Ok(Action::ScrollRight),
        KeyCode::Esc => Ok(Action::Quit),
        KeyCode::Backspace => Ok(Action::Noop),
        KeyCode::Enter => Ok(Action::Noop),
        _ => Err(format!("Unsuportted key event {:?}", key_event)),
    }
}

fn get_search_action(key_event: &KeyEvent, search_model: &SearchModel) -> Result<Action, String> {
    match search_model {
        SearchModel::SearchCreation(_)=> {
            match key_event.code {
                KeyCode::Char(c) => Ok(Action::SearchAddKey(c)),
                KeyCode::Enter => Ok(Action::SearchDown),
                KeyCode::Esc => Ok(Action::SearchEnd),
                KeyCode::Backspace => Ok(Action::SearchRemoveKey),
                _ => get_default_mode_action(key_event),
            }
        },
        SearchModel::SearchInProgress(_) => {
            match key_event.code {
                KeyCode::Char('n') => Ok(Action::SearchDown),
                KeyCode::Char('?') => Ok(Action::SearchUp),
                KeyCode::Enter => Ok(Action::SearchDown),
                KeyCode::Esc => Ok(Action::SearchEnd),
                KeyCode::Backspace => Ok(Action::SearchRemoveKey),
                _ => get_default_mode_action(key_event),
            }
        },
    }
}

fn get_command_action(key_event: &KeyEvent, command: &String) -> Result<Action, String> {
    match key_event.code {
        KeyCode::Char(c) => Ok(Action::CommandAddKey(c)),
        KeyCode::Enter => Ok(Action::CommandExecute(command.clone())),
        KeyCode::Esc => Ok(Action::CommandEnd),
        KeyCode::Backspace => Ok(Action::CommandRemoveKey),
        _ => get_default_mode_action(key_event),
    }
}


fn get_table_action(data_model: &DataModel) -> Result<Action, String> {
    let event = event::read().map_err(convert_to_display_error)?;
    let key_event = match event {
        Event::Key(key_event) => key_event,
        _ => { return Err(format!("Unsupported event {:?}", event)); }
    };
    match &data_model.user_model {
        UserModel::DefaultMode => get_default_mode_action(&key_event),
        UserModel::SearchMode(search_model) => get_search_action(&key_event, search_model),
        UserModel::CommandMode(command) => get_command_action(&key_event, command),
    }
}

fn log_action(action: &Action, log_file: &mut Option<File>) -> Result<(), String> {
    match log_file {
        None => Ok(()),
        Some(log_file) => log_file.write_all(format!("{:?}#\n", action).as_bytes()).map_err(convert_to_display_error),
    }
}


fn create_command_paragraph(model: &DataModel) -> Result<Paragraph<'_>, String> {
    let text = match &model.user_model {
        UserModel::DefaultMode => Line::from("..."),
        UserModel::SearchMode(search_model) => match search_model {
            SearchModel::SearchCreation(search_string) => Line::from(format!("/{}", search_string)),
            SearchModel::SearchInProgress(search_string) => Line::from(format!("/{}", search_string)),
        },
        UserModel::CommandMode(command_string) => Line::from(format!(":{}", command_string)),
    };
    Ok(Paragraph::new(vec!(text)))
}

fn get_table_begin_row(model: &DataFrameModel, widget_height: i64) -> i64 {
    let result = model.current_position.row - (widget_height / 2);
    let result = cmp::min(result, model.total_num_rows - widget_height);
    let result = cmp::max(result, 0);
    result
}

fn render_data_frame_table(model: &DataFrameModel, area: &Rect, frame: &mut Frame, tui_state: &mut TuiState) -> Result<(), String> {
    let raw_height = area.height as i64 - 1;
    let table_row_begin = get_table_begin_row(model, raw_height);
    let table_frame = model.current_frame.clone()
        .slice(table_row_begin, raw_height as u32)
        .cast_all(DataType::String, false)
        .fill_null(lit(""))
        .collect().map_err(convert_polars_error)?;
    
    let num_rows = cmp::min(raw_height as usize, table_frame.height() as usize);
    let schema = model.current_frame.clone().collect_schema().map_err(convert_polars_error)?;
    let num_cols = schema.len();
    debug!("Schema: cols {} rows {}", num_cols, num_rows);
    let columns = table_frame.get_columns();

    debug!("Table {}", table_frame.clone());

    let mut data_as_vector: Vec<Vec<Text>> = columns.iter()
        .map(|column| {
            let chunked_array = column.as_materialized_series()
                .str().map_err(convert_polars_error)?;
            chunked_array.iter()
                .map(|opt_str| {
                    opt_str.ok_or(format!("Option not set"))
                        .map(|s| Text::from(s.to_string()))
                }).collect::<Result<Vec<Text>, String>>()
        }).collect::<Result<Vec<Vec<Text>>, String>>()?;

    if data_as_vector.len() == 0 {
        return Err(format!("Empty data frame, cannot render table"));
    }

    let mut max_lengths: Vec<usize> = vec![];
    debug_assert!(num_cols == data_as_vector.len(), "Inconistent columns");
    for i in 0..data_as_vector.len() {
        debug_assert!(num_rows == data_as_vector[i].len(), "Inconsistent columns");
        let max_len = data_as_vector[i].iter()
            .map(|text| text.width())
            .max().unwrap() + 1;
        let col_name = schema.try_get_at_index(i).map_err(convert_polars_error)?;
        let col_name_len = col_name.0.len() + 1;
        max_lengths.push(cmp::max(max_len, col_name_len));
    }

    let mut num_cols_can_fit = 1;
    let col_index = model.current_position.column as usize;
    let mut width_used = max_lengths[col_index];
    for i in col_index..max_lengths.len() {
        if width_used + max_lengths[i] > area.width as usize {
            break;
        }
        width_used += max_lengths[i];
        num_cols_can_fit += 1;
    }

    let rows = (0..num_rows).map(|row_index| {
        Row::new(
            data_as_vector.iter_mut()
                .skip(col_index)
                .take(num_cols_can_fit)
                .map(|data| data[row_index].clone())
        )
    });

    let table = Table::new(
        rows,
        max_lengths.iter()
            .skip(col_index)
            .take(num_cols_can_fit)
            .map(|len| Constraint::Length(*len as u16))
            .collect::<Vec<Constraint>>()
    ).header(
        Row::new(schema.iter_names().skip(col_index).take(num_cols_can_fit).map(|x| x.as_str()))
    ).row_highlight_style(Color::Green);

    let selected_row = model.current_position.row - table_row_begin;
    tui_state.scroll_bar_state = tui_state.scroll_bar_state.position(model.current_position.row as usize);
    tui_state.table_state.select(Some(selected_row as usize));

    frame.render_stateful_widget(table, *area, &mut tui_state.table_state);

    frame.render_stateful_widget(
        Scrollbar::default().orientation(ScrollbarOrientation::VerticalRight).begin_symbol(None).end_symbol(None),
        area.inner(Margin{vertical:1, horizontal:1}),
        &mut tui_state.scroll_bar_state,
    );
    Ok(())
}

fn render_table(model: &DataModel, area: &Rect, frame: &mut Frame, tui_state: &mut TuiState) -> Result<(), String> {
    match &model.table_model {
        TableModel::Csv(csv) => render_data_frame_table(csv, area, frame, tui_state),
        TableModel::Ipc(ipc) => render_data_frame_table(ipc, area, frame, tui_state),
        TableModel::Parquet(parquet) => render_data_frame_table(parquet, area, frame, tui_state),
        TableModel::IpcStream(ipc) => render_data_frame_table(ipc, area, frame, tui_state),
        TableModel::Hdf5(_) => Err(format!("HDF5 rendering not yet implemented")),
    }
}

fn draw_frame(terminal: &mut DefaultTerminal, tui_state: &mut TuiState, model: &DataModel) -> Result<(), String> {
    let command_text = create_command_paragraph(model).map_err(|e| format!("Error creating command paragraph: {}", e))?;
    terminal.try_draw(|frame| -> Result<(), IOError> {
        let area = frame.area();
        let table_area = Rect{
            x: area.x,
            y: area.y,
            width: area.width,
            height: area.height - 1, // Leave space for the footer
        };
        let footer_area = Rect{
            x: area.x,
            y: area.y + area.height - 1,
            width: area.width,
            height: 1,
        };
        render_table(model, &table_area, frame, tui_state).map_err(|s| IOError::new(ErrorKind::Other, s))?;
        frame.render_widget(command_text, footer_area);
        Ok(())
    }).map_err(convert_to_display_error)?;
    Ok(())
}

fn run_loop(terminal: &mut DefaultTerminal, tui: &mut TuiState, model: &mut DataModel, log_file: &mut Option<File>) -> Result<(), String> {
    loop {
        draw_frame(terminal, tui, model)?;
        let action = get_table_action(&model)?;
        log_action(&action, log_file)?;
        match action {
            Action::Quit => break,
            Action::Noop => {},
            _ => update_state(model, &action)?,
        }
    }
    Ok(())
}

fn get_lazy_frame_total_rows(lf: LazyFrame) -> usize {
    lf.select([len().alias("count")])
        .collect()
        .unwrap()
        .column("count")
        .unwrap()
        .u32()
        .unwrap()
        .get(0)
        .unwrap() as usize
}

fn get_total_rows(model: &DataModel) -> usize {
    match &model.table_model {
        TableModel::Csv(csv) => get_lazy_frame_total_rows(csv.current_frame.clone()),
        TableModel::Ipc(ipc) => get_lazy_frame_total_rows(ipc.current_frame.clone()),
        TableModel::Parquet(parquet) => get_lazy_frame_total_rows(parquet.current_frame.clone()),
        TableModel::IpcStream(ipc) => get_lazy_frame_total_rows(ipc.current_frame.clone()),
        TableModel::Hdf5(_) => 0,
    }
}

fn create_log_file(cli: &Cli) -> Result<Option<File>, String> {
    match &cli.log_dir {
        None => Ok(None),
        Some(log_dir) => {
            let log_file = log_dir.join(Path::new("actions_log.txt"));
            File::create(&log_file).map(|x| Some(x)).map_err(convert_to_display_error)
        }
    }
}

fn init_logger(cli: &Cli) -> Result<(), String> {
    let log_dir = match &cli.log_dir {
        None => return Ok(()),
        Some(dir) => dir.as_path(),
    };
    let log_file_path = log_dir.join(Path::new("debug_log.txt"));
    let created_file = File::create(log_file_path).map_err(convert_to_display_error)?;
    let target = Box::new(created_file);

    env_logger::Builder::new()
        .target(env_logger::Target::Pipe(target))
        .filter(None, LevelFilter::Debug)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        }).init();
    Ok(())
}


fn main() -> Result<(), String> {
    let cli = Cli::parse();
    init_logger(&cli)?;
    let mut model = create_model(&cli)?;
    let mut log_file = create_log_file(&cli)?;
    let mut terminal = ratatui::init();
    let mut tui_state = TuiState{
        scroll_bar_state: ScrollbarState::new(get_total_rows(&model)),
        table_state: TableState::default().with_selected(Some(0)),
    };
    let final_result = run_loop(&mut terminal, &mut tui_state, &mut model, &mut log_file);
    ratatui::restore();
    final_result
}


#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::DataFrame;
    use polars::df;
    use polars_io::prelude::*;
    use tempfile::NamedTempFile;
    // use crate::convert_polars_error;

    fn get_test_df() -> Result<DataFrame, String> {
        let df = df! [
            "column1" => [1, 2, 3],
            "column2" => [4, 5, 6],
            "column3" => [7, 8, 9]
        ];
        df.map_err(convert_polars_error)
    }

    #[test]
    fn test_csv_load() -> Result<(), String> {
        let mut test_df = get_test_df()?;

        let mut file = NamedTempFile::new().map_err(convert_to_display_error)?;
        let output_path = file.path().to_path_buf();

        CsvWriter::new(&mut file)
            .include_header(true)
            .with_separator(b',')
            .finish(&mut test_df).map_err(convert_polars_error)?;

        let mut file_type = Some("csv".to_string());
        let table_model = create_table_model(&output_path,  &file_type)?;

        let data_model = match &table_model {
            TableModel::Csv(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(data_model.total_num_rows, 3);
        assert_eq!(data_model.total_num_columns, 3);

        file_type = None;
        let unkownn_model = create_table_model(&output_path, &file_type)?;
        let unknown_data_model = match &unkownn_model {
            TableModel::Csv(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(unknown_data_model.total_num_rows, 3);
        assert_eq!(unknown_data_model.total_num_columns, 3);


        Ok(())
    }


    #[test]
    fn test_ipc_load() -> Result<(), String> {
        let mut test_df = get_test_df()?;

        let mut file = NamedTempFile::new().map_err(convert_to_display_error)?;
        let output_path = file.path().to_path_buf();

        IpcWriter::new(&mut file)
            .finish(&mut test_df).map_err(convert_polars_error)?;

        let mut file_type = Some("ipc".to_string());
        let table_model = create_table_model(&output_path,  &file_type)?;

        let data_model = match &table_model {
            TableModel::Ipc(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(data_model.total_num_rows, 3);
        assert_eq!(data_model.total_num_columns, 3);

        file_type = None;
        let unkownn_model = create_table_model(&output_path, &file_type)?;
        let unknown_data_model = match &unkownn_model {
            TableModel::Ipc(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(unknown_data_model.total_num_rows, 3);
        assert_eq!(unknown_data_model.total_num_columns, 3);

        Ok(())
    }

    #[test]
    fn test_parquet_load() -> Result<(), String> {
        let mut test_df = get_test_df()?;

        let mut file = NamedTempFile::new().map_err(convert_to_display_error)?;
        let output_path = file.path().to_path_buf();

        ParquetWriter::new(&mut file)
            .finish(&mut test_df).map_err(convert_polars_error)?;

        let mut file_type = Some("parquet".to_string());
        let table_model = create_table_model(&output_path,  &file_type)?;

        let data_model = match &table_model {
            TableModel::Parquet(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(data_model.total_num_rows, 3);
        assert_eq!(data_model.total_num_columns, 3);

        file_type = None;
        let unkownn_model = create_table_model(&output_path, &file_type)?;
        let unknown_data_model = match &unkownn_model {
            TableModel::Parquet(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(unknown_data_model.total_num_rows, 3);
        assert_eq!(unknown_data_model.total_num_columns, 3);

        Ok(())
    }

    #[test]
    fn test_ipc_stream_load() -> Result<(), String> {
        let mut test_df = get_test_df()?;

        let mut file = NamedTempFile::new().map_err(convert_to_display_error)?;
        let output_path = file.path().to_path_buf();

        IpcStreamWriter::new(&mut file)
            .finish(&mut test_df).map_err(convert_polars_error)?;

        let mut file_type = Some("ipc_stream".to_string());
        let table_model = create_table_model(&output_path,  &file_type)?;

        let data_model = match &table_model {
            TableModel::IpcStream(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(data_model.total_num_rows, 3);
        assert_eq!(data_model.total_num_columns, 3);

        file_type = None;
        let unkownn_model = create_table_model(&output_path, &file_type)?;
        let unknown_data_model = match &unkownn_model {
            TableModel::IpcStream(csv) => Ok(csv),
            _ => Err("Wrong file type".to_string()),
        }?;

        assert_eq!(unknown_data_model.total_num_rows, 3);
        assert_eq!(unknown_data_model.total_num_columns, 3);

        Ok(())
    }
}










