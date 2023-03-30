//+------------------------------------------------------------------+
//|                                                    structure.mqh |
//|                                       Copyright 2022, Hedge Ltd. |
//|                                            https://www.hedge.com |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2022, Hedge Ltd."
#property link          "https://www.hedge.com"
#property description   "header for custom struct"
#property version       "1.00"
//+--------------------------------------------------------------------------------------------------------------+

//ArraySetAsSeries(rates, false); !!!!!!!!!!!!!!!

struct MyMqlRates
{
   datetime time;                // Period start time
   double   open;                // Open price
   double   high;                // The highest price of the period
   double   low;                 // The lowest price of the period
   double   close;               // Close price
   int      tick_volume;         // Tick volume
   double   spread;              // Spread
   int      ticks_closeHigh;     // Number of ticks between high and close.
   int      ticks_closeLow;      // Number of ticks between low and close.
   
   double   variation_closeOpen; // Variation between close price of ith candle and open price of jth candle
   double   variation_closeHigh; // Variation between high price of ith candle and close price of jth candle
   double   variation_closeLow;  // Variation between low price of ith candle and close price of jth candle
   
   double   average_price;       //Average opening, high, lown and close price
   
   int      volume_profile;      //Volume profile
};

//+--------------------------------------------------------------------------------------------------------------+