"""
HW-6 Data Table

NAME: Dominick Schulz
DATE: Fall 2023
CLASS: CPSC 322

"""

import csv
import tabulate


class DataRow:
    """A basic representation of a relational table row. The row maintains
    its corresponding column information.

    """
    
    def __init__(self, columns=[], values=[]):
        """Create a row from a list of column names and data values.
           
        Args:
            columns: A list of column names for the row
            values: A list of the corresponding column values.

        Notes: 
            The column names cannot contain duplicates.
            There must be one value for each column.

        """
        if not isinstance(values, list):
            raise ValueError("values must be a list")   
                    
                
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        if len(columns) != len(values):
            raise ValueError('mismatched number of columns and values')
        self.__columns = columns.copy()
        self.__values = values.copy()


        
    def __repr__(self):
        """Returns a string representation of the data row (formatted as a
        table with one row).

        Notes: 
            Uses the tabulate library to pretty-print the row.

        """
        return tabulate.tabulate([self.values()], headers=self.columns())

        
    def __getitem__(self, column):
        """Returns the value of the given column name.
        
        Args:
            column: The name of the column.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        return self.values()[self.columns().index(column)]
    


    def __setitem__(self, column, value):
        """Modify the value for a given row column.
        
        Args: 
            column: The column name.
            value: The new value.

        """
        if column not in self.columns():
            raise IndexError('bad column name')
        self.__values[self.columns().index(column)] = value


    def __delitem__(self, column):
        """Removes the given column and corresponding value from the row.

        Args:
            column: The column name.

        """

        if column in self.__columns:
            index = self.__columns.index(column)
            del self.__columns[index]
            del self.__values[index]
        else:
            raise IndexError('bad column name')



    def __eq__(self, other):
        """Returns true if this data row and other data row are equal.
           
        Args:
            other: The other row to compare this row to.

        Notes:
            Checks that the rows have the same columns and values.

        """
        
        if not isinstance(other, DataRow):
            return False
        
        elif self.__columns == other.__columns and self.__values == other.__values:
            return True
        else:
            return False
    

    
    def __add__(self, other):
        """Combines the current row with another row into a new row.
        
        Args:
            other: The other row being combined with this one.

        Notes:
            The current and other row cannot share column names.

        """
        if not isinstance(other, DataRow):
            raise ValueError('expecting DataRow object')
        if len(set(self.columns()).intersection(other.columns())) != 0:
            raise ValueError('overlapping column names')
        return DataRow(self.columns() + other.columns(),
                       self.values() + other.values())
        


    def columns(self):
        """Returns a list of the columns of the row."""
        return self.__columns.copy()


    def values(self, columns=None):
        """Returns a list of the values for the selected columns in the order
        of the column names given.
           
        Args:
            columns: The column values of the row to return. 

        Notes:
            If no columns given, all column values returned.

        """
        if columns is None:
            return self.__values.copy()
        if not set(columns) <= set(self.columns()):
            raise ValueError('duplicate column names')
        return [self[column] for column in columns]


##################################################################################


    def select(self, columns=None):
        """Returns a new data row for the selected columns in the order of the
        column names given.

        Args:
            columns: The column values of the row to include.
        
        Notes:
            If no columns given, all column values included.

        """        
        
        if columns == None:
            copy_data_row = DataRow(self.columns(), self.values())
            return copy_data_row
        
        if all(col in self.columns() for col in columns):
            new_row = DataRow()
            for col in columns:
                index = self.__columns.index(col)
                temp_row = DataRow([self.__columns[index]],[self.__values[index]])
                new_row = (new_row + temp_row)
            return new_row
        else:
            raise ValueError('bad column name')  

    
    def copy(self):
        """Returns a copy of the data row."""
        return self.select()

    

class DataTable:
    """A relational table consisting of rows and columns of data.

    Note that data values loaded from a CSV file are automatically
    converted to numeric values.

    """
    
    def __init__(self, columns=[]):
        """Create a new data table with the given column names

        Args:
            columns: A list of column names. 

        Notes:
            Requires unique set of column names. 

        """
        if len(columns) != len(set(columns)):
            raise ValueError('duplicate column names')
        self.__columns = columns.copy()
        self.__row_data = []



    def __repr__(self):
        """Return a string representation of the table.
        
        Notes:
            Uses tabulate to pretty print the table.

        """  
        
        return tabulate.tabulate([row.values() for row in self.__row_data], headers=self.columns())
        

    
    def __getitem__(self, row_index):
        """Returns the row at row_index of the data table.
        
        Notes:
            Makes data tables iterable over their rows.

        """

        #     return DataRow(self.__columns, list(self.__row_data[row_index]))
        
        return self.__row_data[row_index]


    
    def __delitem__(self, row_index):
        """Deletes the row at row_index of the data table.

        """
        if row_index < 0 or row_index >= len(self.__row_data):
            raise IndexError('Index out of range') 
        else:
            del self.__row_data[row_index]

        
    def load(self, filename, delimiter=','):
        """Add rows from given filename with the given column delimiter.

        Args:
            filename: The name of the file to load data from
            delimeter: The column delimiter to use

        Notes:
            Assumes that the header is not part of the given csv file.
            Converts string values to numeric data as appropriate.
            All file rows must have all columns.
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            num_cols = len(self.columns())
            for row in reader:
                row_cols = len(row)                
                if num_cols != row_cols:
                    raise ValueError(f'expecting {num_cols}, found {row_cols}')
                converted_row = []
                for value in row:
                    converted_row.append(DataTable.convert_numeric(value.strip()))
                self.__row_data.append(DataRow(self.columns(), converted_row))

                    
    def save(self, filename, delimiter=','):
        """Saves the current table to the given file.
        
        Args:
            filename: The name of the file to write to.
            delimiter: The column delimiter to use. 

        Notes:
            File is overwritten if already exists. 
            Table header not included in file output.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter, quotechar='"',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in self.__row_data:
                writer.writerow(row.values())


    def column_count(self):
        """Returns the number of columns in the data table."""
        return len(self.__columns)


    def row_count(self):
        """Returns the number of rows in the data table."""
        return len(self.__row_data)


    def columns(self):
        """Returns a list of the column names of the data table."""
        return self.__columns.copy()


    def append(self, row_values):
        """Adds a new row to the end of the current table. 

        Args:
            row_data: The row to add as a list of values.
        
        Notes:
            The row must have one value per column. 
        """
    

        if len(row_values) != len(self.columns()):
            raise ValueError('mismatched number of columns in row_values')
        temp_row = DataRow(self.columns(), row_values)
        self.__row_data.append(temp_row)


    
    def rows(self, row_indexes):
        """Returns a new data table with the given list of row indexes. 

        Args:
            row_indexes: A list of row indexes to copy into new table.
        
        Notes: 
            New data table has the same column names as current table.

        """
        
        new_table = DataTable(self.__columns)
    
        for index in row_indexes:
            if index < 0 or index >= len(self.__row_data):
                raise IndexError('Out of range index')

            if len(self.__columns) != len(new_table.columns()):
                raise ValueError('Mismatched number of columns in row_values')
            new_table.__row_data.append(self.__row_data[index])
        return new_table 
        
        
    def drop(self, columns):
        """Removes the given columns from the current table.
        Args:
        column: the name of the columns to drop
        """
        
        if not isinstance(columns, list):
            raise ValueError('Column(s) not found in column names.')
        
        for col in columns:
            if col in self.columns():
                col_index = self.columns().index(col)
                for row in self.__row_data:
                    del row[col]                    
                del self.__columns[col_index]
    

    
    def copy(self):
        """Returns a copy of the current table."""
        table = DataTable(self.columns())
        for row in self:
            table.append(row.values())
        return table
    

    def update(self, row_index, column, new_value):
        """Changes a column value in a specific row of the current table.

        Args:
            row_index: The index of the row to update.
            column: The name of the column whose value is being updated.
            new_value: The row's new value of the column.

        Notes:
            The row index and column name must be valid. 

        """
        if row_index < 0 or row_index >= len(self.__row_data):
            raise IndexError('Out of range index')
        if column not in self.__columns:
            raise IndexError('invalid column name')
        
        self.__row_data[row_index][column] = new_value
        
        

    @staticmethod
    def combine(table1, table2, columns=[], non_matches=False):
        """Returns a new data table holding the result of combining table 1 and 2.

        Args:
            table1: First data table to be combined.
            table2: Second data table to be combined.
            columns: List of column names to combine on.
            non_matches: Include non-matches in the answer.

        Notes:
            If columns to combine on are empty, performs all combinations.
            Column names to combine must be in both tables.
            Duplicate column names are removed from the table2 portion of the result.

        """
        
        
        for col in columns:
            if col not in table1.columns() or col not in table2.columns():
                raise IndexError('combine columns do not exist in both tables')
            
        if len(columns) != len(set(columns)):
            raise IndexError('duplicate column names')
        
        #creates a combined columns that will be used in the final data table to be returned
        table1_columns = table1.columns()
        table2_columns = [col for col in table2.columns() if col not in columns]
        combined_columns = table1_columns + table2_columns
        
        
        # removes duplicate columns, but maintains the order that they were originally
        unique_columns = []
        for col in combined_columns:
            if col not in unique_columns:
                unique_columns.append(col)

        combined_columns = unique_columns
        
        #new table is to be returned at the end
        new_table = DataTable(combined_columns)
        
        #table created with only the combine on columns (this is used for counting the total columns in first non match example)
        combine_on_table = DataTable(columns)

        
        
        ##### this portion is for when non matches are included
        if non_matches:
            for r1 in table1:
                for r2 in table2:
                    r1 = DataRow(r1.columns(), r1.values()) 
                                    
                    if r1.select(columns) == r2.select(columns):
                        append_values = r1.values() + r2.select(table2_columns).values()
                        new_table.append(append_values)
            
            
            
            # adds non matches to the table along with matches
            for r1 in table1:
                found = False
                for r2 in table2:
                    r1 = DataRow(r1.columns(), r1.values()) 
                                    
                    if r1.select(columns) == r2.select(columns):
                        found = True
                        # print('found is now True')
                if not found:     
                    blanks_num = DataTable(r2.columns()).column_count() - combine_on_table.column_count()
                    
                    actual_blanks = []
                    
                    for i in range(blanks_num):
                        actual_blanks = actual_blanks + ['']
                        
                    
                    append_values = r1.values()
                    append_values.extend(actual_blanks)
                    new_table.append(append_values)

            
            
 
            
            for r1 in table2:
                found = False
                for r2 in table1:
                    r1 = DataRow(r1.columns(), r1.values()) 
                                    
                    if r1.select(columns) == r2.select(columns):
                        found = True
                        
                if not found:
                    
                    #creates row that will be added to the table
                    append_values = []
                    for col in new_table.columns():
                        
                        #tries to add the value, but if index error is thrown (meaning there is no value in the column), it will add a blank in the except
                        try:
                            append_values.extend([r1[col]])
                        except IndexError:
                            append_values.extend([''])
            
            #adds the append values row to the table
            new_table.append(append_values)

        
        ##### This portion is for when non matches are not included
        if not non_matches:
            for r1 in table1:
                for r2 in table2:
                    r1 = DataRow(r1.columns(), r1.values()) 
                                    
                    if r1.select(columns) == r2.select(columns):
                        append_values = r1.values() + r2.select(table2_columns).values()
                                                
                        new_table.append(append_values)
                        
        return new_table
        

    
    @staticmethod
    def convert_numeric(value):
        """Returns a version of value as its corresponding numeric (int or
        float) type as appropriate.

        Args:
            value: The string value to convert

        Notes:
            If value is not a string, the value is returned.
            If value cannot be converted to int or float, it is returned.

         """
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        else:
            return value
