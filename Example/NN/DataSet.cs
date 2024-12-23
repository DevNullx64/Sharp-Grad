﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.NN
{
    public static class DataSet
    {
        public const int N = 30; // TODO: Rename it to a more meaningful name.

        public class Data(List<int> X, List<int> Y)
        {
            public readonly List<int> X = X;
            public readonly List<int> Y = Y;
        }

        public static List<Data> GetDataSet(int n)
        {
            var rand = new Random();
            List<Data> dataset = [];
            for (int i = 0; i < n; i++)
            {
                int x = rand.Next(-15, 15);
                int y = rand.Next(-15, 15);
                List<int> X = [x, y];
                List<int> Y = [(y > 2 * x + 5) ? 1 : 2];

                dataset.Add(new(X, Y));
            }
            return dataset;
        }

        private static readonly string UpperRow = "╔" + new string('═', N) + "╗";
        private static readonly string LowerRow = "╚" + new string('═', N) + "╝";

        public static int[,] GetMat(List<Data> v)
        {
            int[,] mat = new int[N, N];
            for (int i = 0; i < v.Count; i++)
                mat[v[i].X[0] + 15, v[i].X[1] + 15] = v[i].Y[0];
            return mat;
        }

        public static void Scatter(List<Data> x, List<Data> y)
        {
            int[,] matX = GetMat(x);
            int[,] matY = GetMat(y);

            Console.Write(UpperRow);
            Console.WriteLine(UpperRow);

            for (int r = 0; r < N; r++)
            {
                Console.Write("║");
                for (int c = 0; c < N; c++)
                {
                    switch (matX[r, c])
                    {
                        case 0:
                            Console.Write(" ");
                            break;
                        case 1:
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.Write("o");
                            break;
                        case 2:
                            Console.ForegroundColor = ConsoleColor.Blue;
                            Console.Write("o");
                            break;
                    }
                    Console.ResetColor();
                }
                Console.Write("║║");
                for (int j = 0; j < N; j++)
                {
                    switch (matY[r, j])
                    {
                        case 0:
                            Console.Write(" ");
                            break;
                        case 1:
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.Write("o");
                            break;
                        case 2:
                            Console.ForegroundColor = ConsoleColor.Blue;
                            Console.Write("o");
                            break;
                    }
                    Console.ResetColor();
                }
                Console.WriteLine("║");
            }

            Console.Write(LowerRow);
            Console.WriteLine(LowerRow);
        }
    }
}
