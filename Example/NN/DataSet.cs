using System;
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
                List<int> X = new();
                List<int> Y = new();
                X.Add(rand.Next(-15, 15));
                X.Add(rand.Next(-15, 15));
                int x = X[0];
                int y = X[1];
                if (y > 2 * x + 5)
                    Y.Add(1);
                else
                    Y.Add(2);

                dataset.Add(new(X, Y));
            }
            return dataset;
        }
        
        public static void Scatter(List<Data> v)
        {
            Console.Clear();
            Console.WriteLine("\x1b[3J");

            int[,] mat = new int[300, 300];
            for (int i = 0; i < v.Count; i++)
            {
                mat[v[i].X[0] + 15, v[i].X[1] + 15] = v[i].Y[0];
            }
            Console.Write("╔");
            for (int i = 0; i < 30; i++)
            {
                Console.Write("═");
            }
            Console.WriteLine("╗");
            for (int i = 0; i < 30; i++)
            {
                Console.Write("║");
                for (int j = 0; j < 30; j++)
                {
                    switch (matX[r, c])
                    {
                        Console.Write(" ");
                    }
                    if (mat[i, j] == 1)
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
