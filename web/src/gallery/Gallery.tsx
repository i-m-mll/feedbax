import { installTooltips } from '@window-shell/browser/tooltip';
import {
  Box, ChartLine, ChevronDown, FileText, FlaskConical, Maximize, Plus, Search, Settings, ZoomIn, ZoomOut,
} from 'lucide-react';
import { useLayoutEffect, useState, type ReactNode } from 'react';
import { setAppearance } from '@/appearance/appearance';
import { useRootAttribute } from '@/appearance/useRootAttribute';
import { Button, IconButton } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { Combobox } from '@/components/ui/Combobox';
import { Dialog } from '@/components/ui/Dialog';
import { ListRow } from '@/components/ui/ListRow';
import {
  ContextMenu, ContextMenuContent, ContextMenuTrigger, Menu, MenuCheckboxItem, MenuContent, MenuGroup, MenuGroupLabel, MenuItem,
  MenuSeparator, MenuTrigger,
} from '@/components/ui/Menu';
import { NumberField } from '@/components/ui/NumberField';
import { PanelHeader } from '@/components/ui/PanelHeader';
import { Popover, PopoverClose, PopoverContent, PopoverTrigger } from '@/components/ui/Popover';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Slider } from '@/components/ui/Slider';
import { Select, type Option } from '@/components/ui/Select';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { TextField } from '@/components/ui/TextField';

type Theme = 'light' | 'dark';
type Density = 'compact' | 'regular';

const kinds = [
  { value: 'train', label: 'Train', glyph: <Box /> },
  { value: 'evaluate', label: 'Evaluate', glyph: <FlaskConical /> },
  { value: 'analyze', label: 'Analyze', glyph: <ChartLine /> },
  { value: 'report', label: 'Report', glyph: <FileText />, disabled: true },
] as const satisfies readonly Option<string>[];

const optimizers: Option<string>[] = [
  'adam', 'adamw', 'adabelief', 'adagrad', 'lamb', 'lion', 'nadam', 'radam', 'rmsprop', 'sgd',
].map((value) => ({ value, label: value }));

function Section({ name, children }: { name: string; children: ReactNode }) {
  return (
    <section className="grid gap-2" aria-label={name}>
      <h2 className="text-sm font-medium text-shell-muted">{name}</h2>
      <div className="flex flex-wrap items-center gap-3">{children}</div>
    </section>
  );
}

function State({ name, children }: { name: string; children: ReactNode }) {
  return (
    <div className="grid justify-items-start gap-1">
      {children}
      <span className="text-xs text-shell-muted">{name}</span>
    </div>
  );
}

export function Gallery() {
  useLayoutEffect(() => installTooltips(), []);
  const theme = useRootAttribute<Theme>('data-theme');
  const density = useRootAttribute<Density>('data-density');
  const [text, setText] = useState('policy');
  const [number, setNumber] = useState<number | null>(0.001);
  const [position, setPosition] = useState(0.6);
  const [checked, setChecked] = useState(true);
  const [kind, setKind] = useState<string | null>('train');
  const [optimizer, setOptimizer] = useState<string | null>('adam');
  const [windowState, setWindowState] = useState('both');
  const [library, setLibrary] = useState('library');
  const [expanded, setExpanded] = useState(true);
  const [dialog, setDialog] = useState(false);
  const [chosen, setChosen] = useState('');

  return (
    <main className="min-h-screen bg-shell-surface text-shell-foreground">
      <header className="sticky top-0 z-10 flex h-panel-header items-center gap-3 border-b border-shell-divider bg-shell-chrome px-panel-header-inline">
        <h1 className="flex-1 font-medium">Controls</h1>
        <SegmentedControl<Theme> aria-label="Theme" value={theme} onValueChange={(value) => setAppearance(window, 'theme', value)}
          items={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }]} />
        <SegmentedControl<Density> aria-label="Density" value={density}
          onValueChange={(value) => setAppearance(window, 'density', value)}
          items={[{ value: 'compact', label: 'Compact' }, { value: 'regular', label: 'Regular' }]} />
      </header>
      <div className="grid max-w-5xl gap-6 p-6">
        <Section name="Button">
          <State name="default"><Button>Train</Button></State>
          <State name="primary"><Button variant="primary">Run</Button></State>
          <State name="ghost"><Button variant="ghost">Cancel</Button></State>
          <State name="glyph"><Button><Plus />Add</Button></State>
          <State name="disabled"><Button disabled>Train</Button></State>
          <State name="primary disabled"><Button variant="primary" disabled>Run</Button></State>
          <State name="pressed"><Button variant="ghost" pressed>Outline</Button></State>
          <State name="start"><Button variant="ghost" align="start" className="w-40">GRU</Button></State>
          <State name="mark"><Button size="mark">3</Button></State>
        </Section>

        <Section name="IconButton">
          <State name="ghost"><IconButton label="Fit" icon={<Maximize />} /></State>
          <State name="default"><IconButton variant="default" label="Settings" icon={<Settings />} /></State>
          <State name="pressed"><IconButton label="Zoom in" icon={<ZoomIn />} pressed /></State>
          <State name="disabled"><IconButton label="Zoom out" icon={<ZoomOut />} disabled /></State>
          <State name="mark"><IconButton size="mark" label="Add Port" icon={<Plus />} /></State>
        </Section>

        <Section name="TextField">
          <State name="default"><TextField aria-label="Name" value={text} onValueChange={setText} /></State>
          <State name="placeholder"><TextField aria-label="Search" placeholder="Search" leading={<Search />} /></State>
          <State name="invalid"><TextField aria-label="Invalid" defaultValue="2x" invalid /></State>
          <State name="read-only"><TextField aria-label="Read-only" value="checkpoint-12" readOnly /></State>
          <State name="disabled"><TextField aria-label="Disabled" value="policy" disabled /></State>
        </Section>

        <Section name="NumberField">
          <State name="default"><NumberField aria-label="Learning rate" value={number} onValueChange={setNumber} step={0.001} /></State>
          <State name="bounded"><NumberField aria-label="Batch size" value={64} onValueChange={() => undefined} min={1} max={64} /></State>
          <State name="invalid"><NumberField aria-label="Epochs" value={-1} onValueChange={() => undefined} invalid /></State>
          <State name="read-only"><NumberField aria-label="Seed" value={7} onValueChange={() => undefined} readOnly /></State>
          <State name="disabled"><NumberField aria-label="Steps" value={100} onValueChange={() => undefined} disabled /></State>
        </Section>

        <Section name="Slider">
          <State name="default"><Slider aria-label="Position" value={position} onValueChange={setPosition} min={0} max={1} step={0.01} className="w-48" /></State>
          <State name="disabled"><Slider aria-label="Locked" value={0.3} onValueChange={() => undefined} min={0} max={1} step={0.01} disabled className="w-48" /></State>
        </Section>
        <Section name="Checkbox">
          <State name="checked"><Checkbox label="Shuffle" checked={checked} onCheckedChange={setChecked} /></State>
          <State name="unchecked"><Checkbox label="Clip" checked={false} onCheckedChange={() => undefined} /></State>
          <State name="indeterminate"><Checkbox label="All" checked={false} indeterminate onCheckedChange={() => undefined} /></State>
          <State name="disabled"><Checkbox label="Resume" checked disabled onCheckedChange={() => undefined} /></State>
          <State name="unlabelled"><Checkbox aria-label="Select row" checked onCheckedChange={() => undefined} /></State>
        </Section>

        <Section name="Select">
          <State name="value"><Select aria-label="Kind" items={kinds} value={kind} onValueChange={setKind} className="w-40" /></State>
          <State name="placeholder"><Select aria-label="Kind" items={kinds} value={null} placeholder="Kind" onValueChange={() => undefined} className="w-40" /></State>
          <State name="disabled"><Select aria-label="Kind" items={kinds} value="train" disabled onValueChange={() => undefined} className="w-40" /></State>
        </Section>

        <Section name="Combobox">
          <State name="value"><Combobox aria-label="Optimizer" items={optimizers} value={optimizer} onValueChange={setOptimizer} className="w-48" /></State>
          <State name="placeholder"><Combobox aria-label="Optimizer" items={optimizers} value={null} placeholder="Optimizer" onValueChange={() => undefined} className="w-48" /></State>
          <State name="invalid"><Combobox aria-label="Optimizer" items={optimizers} value={null} invalid onValueChange={() => undefined} className="w-48" /></State>
          <State name="disabled"><Combobox aria-label="Optimizer" items={optimizers} value="sgd" disabled onValueChange={() => undefined} className="w-48" /></State>
        </Section>

        <Section name="Menu">
          <State name={chosen || 'closed'}>
            <Menu>
              <MenuTrigger render={<Button />}><Plus />Step<ChevronDown /></MenuTrigger>
              <MenuContent>
                {kinds.map((item) => (
                  <MenuItem key={item.value} glyph={item.glyph} disabled={'disabled' in item}
                    onClick={() => setChosen(item.label)}>{item.label}</MenuItem>
                ))}
                <MenuSeparator />
                <MenuGroup>
                  <MenuGroupLabel>Map</MenuGroupLabel>
                  <MenuItem shortcut="⇧1" onClick={() => setChosen('Fit')}>Fit</MenuItem>
                </MenuGroup>
                <MenuSeparator />
                <MenuCheckboxItem checked={chosen !== 'Hide grid'} onCheckedChange={(shown) => setChosen(shown ? '' : 'Hide grid')}>
                  Grid
                </MenuCheckboxItem>
              </MenuContent>
            </Menu>
          </State>
          <State name="context">
            <ContextMenu>
              <ContextMenuTrigger className="flex h-[5em] w-[12em] items-center justify-center rounded-control border border-dashed border-shell-control text-shell-muted">
                Map
              </ContextMenuTrigger>
              <ContextMenuContent>
                {kinds.map((item) => (
                  <MenuItem key={item.value} glyph={item.glyph} disabled={'disabled' in item}
                    onClick={() => setChosen(item.label)}>{item.label}</MenuItem>
                ))}
              </ContextMenuContent>
            </ContextMenu>
          </State>
        </Section>

        <Section name="Popover">
          <Popover>
            <PopoverTrigger render={<Button />}>Sweep</PopoverTrigger>
            <PopoverContent aria-label="Sweep" className="grid w-64 gap-2">
              <NumberField aria-label="Trials" value={8} onValueChange={() => undefined} />
              <PopoverClose render={<Button variant="primary" />}>Apply</PopoverClose>
            </PopoverContent>
          </Popover>
        </Section>

        <Section name="SegmentedControl">
          <State name="text">
            <SegmentedControl aria-label="Panel" value={library} onValueChange={setLibrary}
              items={[{ value: 'library', label: 'Library' }, { value: 'outline', label: 'Outline' }]} />
          </State>
          <State name="glyph">
            <SegmentedControl aria-label="Window" value={windowState} onValueChange={setWindowState}
              items={[
                { value: 'map', label: 'Map', icon: <Box /> },
                { value: 'both', label: 'Both', icon: <ChartLine /> },
                { value: 'details', label: 'Details', icon: <FileText />, disabled: true },
              ]} />
          </State>
          <State name="disabled">
            <SegmentedControl aria-label="Disabled" value="library" onValueChange={() => undefined} disabled
              items={[{ value: 'library', label: 'Library' }, { value: 'outline', label: 'Outline' }]} />
          </State>
        </Section>

        <Section name="ListRow">
          <div className="w-72 rounded-control border border-shell-divider p-1" role="tree" aria-label="Steps">
            <ListRow role="treeitem" tabIndex={0} name="Pipeline" glyph={<Box />} expanded={expanded} onExpandedChange={setExpanded} trailing="3" />
            {expanded ? <>
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Train" glyph={<Box />} selected trailing="running" />
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Evaluate" glyph={<FlaskConical />} trailing="done" />
              <ListRow role="treeitem" tabIndex={0} depth={1} name="Report" glyph={<FileText />} disabled />
            </> : null}
          </div>
        </Section>

        <Section name="Table">
          <div className="w-full max-w-xl overflow-hidden rounded-control border border-shell-divider">
            <Table>
              <thead><tr><TableHeaderCell>Run</TableHeaderCell><TableHeaderCell>Seed</TableHeaderCell><TableHeaderCell>Loss</TableHeaderCell></tr></thead>
              <tbody>
                <TableRow tabIndex={0}><TableCell>run-1</TableCell><TableCell>1</TableCell><TableCell className="tabular-nums">0.412</TableCell></TableRow>
                <TableRow tabIndex={0} selected><TableCell>run-2</TableCell><TableCell>2</TableCell><TableCell className="tabular-nums">0.388</TableCell></TableRow>
                <TableRow tabIndex={0} related><TableCell>run-3</TableCell><TableCell>2</TableCell><TableCell className="tabular-nums">0.401</TableCell></TableRow>
              </tbody>
            </Table>
          </div>
        </Section>

        <Section name="PanelHeader">
          <div className="w-80 rounded-control border border-shell-divider">
            <PanelHeader title="Details" glyph={<FileText />}
              actions={<><IconButton label="Fit" icon={<Maximize />} /><IconButton label="Settings" icon={<Settings />} /></>} />
          </div>
        </Section>

        <Section name="Dialog">
          <Button onClick={() => setDialog(true)}>Rename</Button>
          {dialog ? (
            <Dialog ariaLabel="Rename" manageFocus onClose={() => setDialog(false)} panelClassName="grid w-72 gap-3">
              <TextField aria-label="Name" defaultValue="policy" data-dialog-autofocus />
              <div className="flex justify-end gap-2">
                <Button variant="ghost" onClick={() => setDialog(false)}>Cancel</Button>
                <Button variant="primary" onClick={() => setDialog(false)}>Rename</Button>
              </div>
            </Dialog>
          ) : null}
        </Section>
      </div>
    </main>
  );
}
