import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from yahoo_fin.stock_info import *
import datetime
import pandas as pd
import os
import time
import numpy as np
from common.CXmlToDict import XmlConvertor


class Parser:
    def __init__(self,
                 config_xml=r'./reporter_config.xml',
                 str_timestamps='1970_01_01_00_00_00'):

        dict_config = XmlConvertor().xml2dict(fpath=config_xml)
        self.output_report_location = eval(dict_config['reporter_session']['metadata']['output_report_location'])
        self.dict_report_paths = dict_config['reporter_session']['raw_reports']
        self.timestamps = str_timestamps
        self.save_data_to_csv = dict_config['reporter_session']['metadata']['save_parsed_data_frame_to_csv']
        self.use_preexisting_ledger = eval(dict_config['reporter_session']['metadata']['use_preexisting_ledger'])
        self.preexisting_ledger_path = eval(dict_config['reporter_session']['metadata']['preexisting_ledger_path'])
        # str_report_folder = os.path.join(self.output_report_location, str_timestamps)
        os.makedirs(self.output_report_location, exist_ok=True)


    def ibkr(self, str_full_file_path_csv_to_parse, b_save_parsed_data_frame_to_csv):
        str_timestamps = self.timestamps
        str_platform_to_prase_from = 'ibkr'

        f_text_handle = open(str_full_file_path_csv_to_parse, 'r')
        lst_trades = []
        d_trades_counter = -1
        for str_line in f_text_handle:
            lst_comma_splitted_line = str_line.strip('\"*\"').strip(';\n').strip('\n').split(',')
            if lst_comma_splitted_line[0].lower() == 'statement':
                pass
            elif lst_comma_splitted_line[0].lower() == 'account information':
                pass
            elif lst_comma_splitted_line[0].lower() == 'net asset value':
                pass
            elif lst_comma_splitted_line[0].lower() == 'change in nav':
                pass
            elif lst_comma_splitted_line[0].lower() == 'mark-to-market performance summary':
                pass
            elif lst_comma_splitted_line[0].lower() == 'realized & unrealized performance summary':
                pass
            elif lst_comma_splitted_line[0].lower() == 'cash report':
                pass
            elif lst_comma_splitted_line[0].lower() == 'open positions':
                pass
            elif lst_comma_splitted_line[0].lower() == 'forex balances':
                pass
            elif lst_comma_splitted_line[0].lower() == 'trades':
                d_trades_counter += 1
                if lst_comma_splitted_line[1].lower() in ['subtotal', 'total'] or \
                        lst_comma_splitted_line[3].lower() not in ['stocks', 'asset category']:
                    continue
                if d_trades_counter > 0:
                    if lst_comma_splitted_line[2].lower() not in ['order']:
                        continue
                    lst_comma_splitted_line[6] = (lst_comma_splitted_line[6] + lst_comma_splitted_line[7]).strip('"')
                    for ii in range(7, len(lst_comma_splitted_line) - 1):
                        lst_comma_splitted_line[ii] = lst_comma_splitted_line[ii + 1]
                    del lst_comma_splitted_line[-1]
                    dt_string = lst_comma_splitted_line[6]
                    str_format = "%Y-%m-%d %H:%M:%S"
                    dt_object = datetime.datetime.strptime(dt_string, str_format)
                    lst_comma_splitted_line[6] = dt_object

                # print(lst_comma_splitted_line)
                # print(str_line)

                lst_trades.append(lst_comma_splitted_line)
            elif lst_comma_splitted_line[0].lower() == 'corporate actions':
                pass
            elif lst_comma_splitted_line[0].lower() == 'deposits & withdrawals':
                pass
            elif lst_comma_splitted_line[0].lower() == 'fees':
                pass
            elif lst_comma_splitted_line[0].lower() == 'dividends':
                pass
            elif lst_comma_splitted_line[0].lower() == 'withholding tax':
                pass
            else:
                continue

        f_text_handle.close()

        pd_trades = pd.DataFrame(lst_trades[1:], columns=lst_trades[0])
        pd_trades.loc[(pd_trades['Symbol'] == 'CFVI'), 'Symbol'] = 'RUM'
        pd_trades.loc[(pd_trades['Symbol'] == 'SMSN'), 'Symbol'] = 'SMSN.IL'
        pd_trades['Quantity'] = pd_trades['Quantity'].astype(int)
        pd_trades['T. Price'] = pd_trades['T. Price'].astype(float)
        pd_trades['C. Price'] = pd_trades['C. Price'].astype(float)
        pd_trades['MTM P/L'] = pd_trades['MTM P/L'].astype(float)
        pd_trades['Proceeds'] = pd_trades['Proceeds'].astype(float)
        pd_trades['Basis'] = pd_trades['Basis'].astype(float)
        pd_trades['Comm/Fee'] = pd_trades['Comm/Fee'].astype(float)
        pd_trades['Realized P/L'] = pd_trades['Realized P/L'].astype(float)

        if b_save_parsed_data_frame_to_csv:
            pd_trades.to_csv(
                fr'.\{str_timestamps}\{str_platform_to_prase_from}_parsed_report_{os.path.basename(str_full_file_path_csv_to_parse)[0:-4]}.csv',
                index=False)
        return pd_trades

    def etrade(self, str_full_file_path_csv_to_parse, b_save_parsed_data_frame_to_csv):
        str_timestamps = self.timestamps
        str_platform_to_prase_from = 'etrade'
        pd_trades = pd.read_csv(str_full_file_path_csv_to_parse)

        # change some columns to conform with the formatting
        pd_trades = pd_trades.rename(columns={'Date Acquired': 'Date/Time',
                                              'Plan Type': 'Asset Category',
                                              'Sellable Qty.': 'Quantity',
                                              'Tax Status': 'Term',
                                              'Record Type': 'DataDiscriminator',
                                              'Grant Number': 'Code'}
                                     )
        # fix time stamp to the format "%d-%b-%Y":
        for index, row in pd_trades.iterrows():
            dic_row = dict(row)
            pd_trades.loc[index, 'Date/Time'] = datetime.datetime.strptime(dic_row['Date/Time'], "%d-%b-%Y")

        # add a purchase value column:
        lst_purchase_value = [0] * len(pd_trades)
        for index, row in pd_trades.iterrows():
            # print(index)
            dic_row = dict(row)
            float_est_market_value_curr = float(
                dic_row['Est. Market Value'].replace('$', '').replace(' ', '').replace(',', ''))
            float_expected_gain_loss = float(
                dic_row['Expected Gain/Loss'].replace('$', '').replace(' ', '').replace(',', ''))
            lst_purchase_value[index] = (float_est_market_value_curr - float_expected_gain_loss) / dic_row['Quantity']
        pd_trades = pd_trades.assign(t_price=pd.Series(lst_purchase_value))
        pd_trades = pd_trades.rename(columns={'t_price': 'T. Price'})
        if b_save_parsed_data_frame_to_csv:
            pd_trades.to_csv(
                fr'.\{str_timestamps}\{str_platform_to_prase_from}_parsed_report_{os.path.basename(str_full_file_path_csv_to_parse)[0:-4]}.csv',
                index=False)
        pd_trades = pd_trades.drop(columns=['Est. Market Value', 'Expected Gain/Loss'])
        return pd_trades

    def binance(self, str_full_file_path_csv_to_parse, b_save_parsed_data_frame_to_csv):
        pass

    def start(self):
        dict_report_paths = self.dict_report_paths
        str_timestamps = self.timestamps
        str_save_data_to_csv = self.save_data_to_csv
        b_use_preexisting_ledger = self.use_preexisting_ledger
        str_preexisting_ledger_path = self.preexisting_ledger_path

        pd_trades = pd.DataFrame(columns=['Trades',
                                          'Header',
                                          'Platform',
                                          'DataDiscriminator',
                                          'Asset Category',
                                          'Currency',
                                          'Symbol',
                                          'Date/Time',
                                          'Quantity',
                                          'Units',
                                          'T. Price',
                                          'C. Price',
                                          'Proceeds',
                                          'Comm/Fee',
                                          'Basis',
                                          'Realized P/L',
                                          'MTM P/L',
                                          'Code',
                                          'Term']
                                 )

        if not b_use_preexisting_ledger:
            print('parsing, ignoring pre-existing ledger file')
            for platform_to_parse_from in dict_report_paths.keys():
                for unparsed_report_path in eval(dict_report_paths[platform_to_parse_from]):
                    print(f'platform_to_parse_from: {platform_to_parse_from}, unparsed_report: {unparsed_report_path}')
                    if 'all' in str_save_data_to_csv.lower():
                        b_save_parsed_data_frame_to_csv = True
                    else:
                        b_save_parsed_data_frame_to_csv = False
                    parser_function_curr = self.__getattribute__(platform_to_parse_from)
                    pd_unparsed_report_curr = parser_function_curr(unparsed_report_path,
                                                                   b_save_parsed_data_frame_to_csv)
                    if platform_to_parse_from.lower() == 'ibkr':
                        pd_unparsed_report_curr.loc[:, 'Units'] = 'Stocks'
                        pd_unparsed_report_curr.loc[:, 'Platform'] = 'IBKR'
                    elif platform_to_parse_from.lower() == 'etrade':
                        pd_unparsed_report_curr.loc[:, 'Trades'] = 'Trades'
                        pd_unparsed_report_curr.loc[:, 'Header'] = 'Data'
                        pd_unparsed_report_curr.loc[:, 'Platform'] = 'ETRADE'
                        pd_unparsed_report_curr.loc[:, 'Units'] = 'Stocks'
                        pd_unparsed_report_curr.loc[:, 'Currency'] = 'USD'
                    elif platform_to_parse_from.lower() == 'binance':
                        pass
                        # pd_unparsed_report_curr.loc[:, 'Trades'] = 'Trades'
                        # pd_unparsed_report_curr.loc[:, 'Header'] = 'Data'
                        # pd_unparsed_report_curr.loc[:, 'Platform'] = 'ETRADE'
                        # pd_unparsed_report_curr.loc[:, 'Units'] = 'Stocks'
                        # pd_unparsed_report_curr.loc[:, 'Currency'] = 'USD'
                    pd_trades = pd.concat([pd_trades, pd_unparsed_report_curr])
        else:
            print(f'using a pre-existing ledger file: {str_preexisting_ledger_path}')
            pd_trades = pd.concat([pd_trades, pd.read_csv(str_preexisting_ledger_path)])

        if 'none' not in str_save_data_to_csv:
            os.makedirs(os.path.join(self.output_report_location, str_timestamps), exist_ok=True)
            pd_trades.to_csv(
                os.path.join(self.output_report_location, str_timestamps, f'ledger_{str_timestamps}.csv'),
                index=False)
        return pd_trades


class Analysis:
    def __init__(self,
                 config_xml=r'.\reporter_config.xml',
                 str_timestamps='1970_01_01_00_00_00'):
        dict_config = XmlConvertor().xml2dict(fpath=config_xml)
        # self.dict_report_paths = dict_config['reporter_session']['raw_reports']
        self.timestamps = str_timestamps
        self.plots = eval(dict_config['reporter_session']['report']['plot_graphs'])
        self.html = eval(dict_config['reporter_session']['report']['html'])
        self.word = eval(dict_config['reporter_session']['report']['word'])
        self.ppt = eval(dict_config['reporter_session']['report']['ppt'])
        self.start_date = eval(dict_config['reporter_session']['report']['start_date'])
        self.end_date = eval(dict_config['reporter_session']['report']['end_date'])
        self.output_report_location = eval(dict_config['reporter_session']['metadata']['output_report_location'])

    def analyzer1(self, pd_trades):
        str_timestamps = self.timestamps
        b_plots = self.plots
        b_html = self.html
        b_word = self.word
        b_ppt = self.ppt
        datetime_start_date = self.start_date
        datetime_end_date = self.end_date
        # str_report_folder = fr'.\{str_timestamps}'
        str_report_folder = os.path.join(self.output_report_location, str_timestamps)
        os.makedirs(self.output_report_location, exist_ok=True)

        if b_plots:
            plt.switch_backend('agg')
            register_matplotlib_converters()
            str_report_folder_plots = os.path.join(str_report_folder, 'figures')
            os.makedirs(str_report_folder_plots, exist_ok=True)

        # analysis:
        int_len_of_symbols = len(pd_trades['Symbol'].unique())
        d_counter = 0
        # lst_pos_analysis = [pd.DataFrame()] * int_len_of_symbols  # + 1)  # +1 for the total
        lst_pos_analysis = [pd.DataFrame()] * (int_len_of_symbols + 1)  # +1 for the total
        for str_curr_ticker_symbol in pd_trades['Symbol'].unique():
            print(f'{d_counter}: {str_curr_ticker_symbol}')
            if b_plots:
                fig = plt.figure(num=d_counter + 1, figsize=[15, 8])
                df_yf_data_curr = get_data(str_curr_ticker_symbol, datetime_start_date, datetime_end_date)
                dt_x_axis_time = df_yf_data_curr.index
                plt.plot(dt_x_axis_time, df_yf_data_curr['close'])
                plt.plot(dt_x_axis_time, df_yf_data_curr['low'], '--', color='red')
                plt.plot(dt_x_axis_time, df_yf_data_curr['high'], '--', color='red')

                for index, row in pd_trades[pd_trades['Symbol'] == str_curr_ticker_symbol].iterrows():
                    dict_row = dict(row)
                    datetime_unique_position_date_curr = \
                        datetime.datetime.strptime(dict_row['Date/Time'], "%Y-%m-%d %H:%M:%S")

                    if (dict_row['DataDiscriminator'] != 'Sell'):
                        str_transaction_type_buy_or_sell = 'buy'
                        str_transaction_type_color = 'blue'
                    else:
                        str_transaction_type_buy_or_sell = 'sell'
                        str_transaction_type_color = 'green'

                    # plot a single instance of transatcation:
                    plt.plot(datetime_unique_position_date_curr,
                             float(dict_row['T. Price']),
                             '.',
                             markersize=20,
                             color=str_transaction_type_color)

                    if 'Cryptocurrency' in dict_row['Asset Category']:
                        pass
                        # plt.text(datetime_unique_position_date_curr,
                        #          float(pd_trades[pd_trades['Date/Time'] == unique_position_date_curr]['T. Price']) * 1.105,
                        #          f'buy x{float(pd_trades[pd_trades["Date/Time"] == unique_position_date_curr]["Quantity"])*100e6:>0.4}S\n'
                        #          f'@ {float(pd_trades[pd_trades["Date/Time"] == unique_position_date_curr]["C. Price"]):>0.2}'
                        #          f'{pd_trades[pd_trades["Symbol"] == str_curr_ticker_symbol]["Currency"].unique()[0]}')
                    else:
                        plt.text(datetime_unique_position_date_curr,
                                 float(dict_row['T. Price']) * 1.105,
                                 f'{str_transaction_type_buy_or_sell} x{int(dict_row["Quantity"])}\n'
                                 f'@ {float(dict_row["T. Price"]):>0.4}')

                plt.plot(dt_x_axis_time, df_yf_data_curr['high'], '--', color='red')
                plt.title(f'{str_curr_ticker_symbol}')
                plt.grid('true', 'both')
                plt.axis([datetime_start_date, datetime_end_date, 0, max(df_yf_data_curr['high'] * 1.1)])

            dict_curr_data = {'Symbol': None,
                              'Current Price': None,
                              'Quantity of Shares': None,
                              'Currency': None,
                              'Break Even Price': None,
                              'Total Monies Spent': None,
                              'Revenue [abs]': None,
                              'Revenue [percent]': None,
                              'Commission': None,
                              'Total Value [abs]': None,
                              'Total Value [percent]': None}
            df_yf_data_curr = get_data(str_curr_ticker_symbol, datetime_start_date, datetime_end_date)

            # currency:
            # str_currency = pd_trades[pd_trades["Symbol"] == str_curr_ticker_symbol]["Currency"].unique()[0]
            str_currency = pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
                                     (pd_trades['DataDiscriminator'] != 'Sell')]['Currency'].unique()[0]

            # current price:
            f_current_price = df_yf_data_curr.close[-1]

            # break even calc
            int_commulative_sum_buy_minus_sell = 0
            float_commulative_scalar_mult_buy_minus_sell = 0
            for index, item in pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol)].iterrows():
                dict_item = dict(item)
                if dict_item['DataDiscriminator'].lower() != 'sell':
                    int_commulative_sum_buy_minus_sell += dict_item['Quantity']
                    float_commulative_scalar_mult_buy_minus_sell += dict_item['Quantity'] * dict_item['T. Price']
                else:
                    int_commulative_sum_buy_minus_sell -= dict_item['Quantity']
                    float_commulative_scalar_mult_buy_minus_sell -= dict_item['Quantity'] * dict_item['T. Price']

            # if 'INTC' == str_curr_ticker_symbol.upper():
            #     print('here')
            # float_break_even_price = \
            #     np.average(pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                          (pd_trades['DataDiscriminator'] != 'Sell')]['T. Price'],
            #                weights=pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                                  (pd_trades['DataDiscriminator'] != 'Sell')]['Quantity'])
            float_break_even_price = float_commulative_scalar_mult_buy_minus_sell / int_commulative_sum_buy_minus_sell

            # plot horizontal lines:
            if b_plots:
                plt.hlines(xmin=datetime_start_date,
                           xmax=datetime_end_date,
                           y=float_break_even_price,
                           colors='blue',
                           linestyles='dashed')
                plt.text(x=datetime_start_date,
                         y=float_break_even_price,
                         s=f'Break Even Price: {float_break_even_price:>0.6}')

            # weighted sum:
            # float_weighted_sum = \
            #     (pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                (pd_trades['DataDiscriminator'] != 'Sell')]['T. Price'] *
            #      pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                (pd_trades['DataDiscriminator'] != 'Sell')]['Quantity']).sum()
            f_total_monies_spent = float_commulative_scalar_mult_buy_minus_sell

            # quantity of shares:
            # int_quantity_of_shares = pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                                    (pd_trades['DataDiscriminator'] != 'Sell')]['Quantity'].sum() - \
            #                          pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol) &
            #                                    (pd_trades['DataDiscriminator'] == 'Sell')]['Quantity'].sum()
            int_quantity_of_shares = int_commulative_sum_buy_minus_sell

            # Comission:
            f_comission = pd_trades[(pd_trades['Symbol'] == str_curr_ticker_symbol)]['Comm/Fee'].sum()

            # Revenue [abs]:
            # f_revenue = int_quantity_of_shares * (f_current_price - float_break_even_price) + f_comission
            f_revenue = (f_current_price - float_break_even_price) * int_quantity_of_shares

            # Revenue [percent]:
            # f_revenue_p = f_revenue / (float_break_even_price + f_comission)
            f_revenue_p = (f_current_price - float_break_even_price) / float_break_even_price * 100

            # Total Value:
            f_total_value = f_current_price * int_quantity_of_shares
            # f_total_value = float_commulative_scalar_mult_buy_minus_sell

            dict_curr_data.update({'Symbol': str_curr_ticker_symbol})
            dict_curr_data.update({'Current Price': f_current_price})
            dict_curr_data.update({'Quantity of Shares': int_quantity_of_shares})
            dict_curr_data.update({'Currency': str_currency})
            dict_curr_data.update({'Break Even Price': float_break_even_price})
            dict_curr_data.update({'Total Monies Spent': f_total_monies_spent})
            dict_curr_data.update({'Revenue [abs]': f_revenue})
            dict_curr_data.update({'Revenue [percent]': f_revenue_p})
            dict_curr_data.update({'Commission': f_comission})
            dict_curr_data.update({'Total Value [abs]': f_total_value})
            lst_pos_analysis[d_counter] = pd.DataFrame(dict_curr_data, index=[d_counter])

            # plots:
            if b_plots:
                textstr = 'Quantity:\n' + \
                          f'Current Price = {f_current_price:>0.6}\n' + \
                          f'Break Even Price = {float_break_even_price:>0.6}{str_currency}\n' + \
                          f'Total Monies Spent = {f_total_monies_spent:>0.6}{str_currency}\n' + \
                          f'Quantity of Shares = {int_quantity_of_shares}\n' + \
                          f'Revenue [abs] = {f_revenue:>0.6}{str_currency}\n' + \
                          f'Total Value [abs] = {f_current_price * int_quantity_of_shares:>0.6}{str_currency}\n'

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                fig.text(0.7, 0.875, textstr, fontsize=14, verticalalignment='top', bbox=props)
                plt.gcf().subplots_adjust(right=0.64)

                # plt.show()
                plt.savefig(os.path.join(str_report_folder_plots, f'{str_curr_ticker_symbol}.png'))
                plt.close()

            d_counter += 1

        # save results summary
        df_aggragated_analyzed_data = pd.concat(lst_pos_analysis)
        # add percentages
        df_aggragated_analyzed_data.loc[:, 'Total Value [percent]'] = \
            df_aggragated_analyzed_data['Total Value [abs]'] / df_aggragated_analyzed_data[
                'Total Value [abs]'].sum() * 100

        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Symbol'] = 'Aggregation'
        df_aggragated_analyzed_data.loc[
            (int_len_of_symbols + 1), 'Current Price'] = None  # df_aggragated_analyzed_data['Current Price'].mean()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Quantity of Shares'] = df_aggragated_analyzed_data[
            'Quantity of Shares'].sum()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Currency'] = 'USD'
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Break Even Price'] = None
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Total Monies Spent'] = df_aggragated_analyzed_data[
            'Total Monies Spent'].sum()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Revenue [abs]'] = df_aggragated_analyzed_data[
            'Revenue [abs]'].sum()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Commission'] = df_aggragated_analyzed_data[
            'Commission'].sum()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Total Value [abs]'] = df_aggragated_analyzed_data[
            'Total Value [abs]'].sum()
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Revenue [percent]'] =\
            float(
                    (df_aggragated_analyzed_data[df_aggragated_analyzed_data['Symbol'] == 'Aggregation']['Total Value [abs]'] /
                     df_aggragated_analyzed_data[df_aggragated_analyzed_data['Symbol'] == 'Aggregation']['Total Monies Spent']
                     - 1) * 100
            )
        df_aggragated_analyzed_data.loc[(int_len_of_symbols + 1), 'Total Value [percent]'] = \
            df_aggragated_analyzed_data['Total Value [percent]'].sum()

        df_aggragated_analyzed_data.to_csv(os.path.join(str_report_folder, 'results.csv'), index=False)
        print(
            f'### analyzer1: printing aggragated results onto: {os.path.join(os.getcwd(), str_report_folder, "results.csv")}')
        return df_aggragated_analyzed_data


if __name__ == '__main__':
    config_xml = r'./reporter_config.xml'
    str_timestamps = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    str_report_folder = fr'.\{str_timestamps}'

    # os.makedirs(str_report_folder, exist_ok=True)

    parser = Parser(str_timestamps=str_timestamps,
                    config_xml=config_xml)

    pd_ledger = parser.start()

    # pd_ledger = pd.read_csv(r'.\ledger.csv')
    # dateZero = datetime.datetime(2020, 1, 1, 0, 0)
    # dateNow = datetime.datetime.now()

    analyzer = Analysis(str_timestamps=str_timestamps,
                        config_xml=config_xml)
    analyzer.analyzer1(pd_trades=pd_ledger)

    # df_analyzed_data = analyzer1(str_timestamps=str_timestamps,
    #                              pd_trades=pd_ledger,
    #                              datetime_start_date=dateZero,
    #                              datetime_end_date=dateNow,
    #                              str_report_folder=str_report_folder,
    #                              b_plots=False)
