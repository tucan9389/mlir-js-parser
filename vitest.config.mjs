import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.{js,ts}'],
    exclude: ['**/node_modules/**', '**/build/**', '**/dist/**'],
    reporters: process.env.CI ? ['dot'] : ['default'],
    testTimeout: 20000,
  },
});
